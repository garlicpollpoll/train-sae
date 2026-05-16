from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from sae_defense.artifact import build_effective_delta, payload_checksum, save_artifact
from sae_defense.config import load_config
from sae_defense.data import load_jsonl
from sae_defense.hooks import ActivationRecorder, resolve_module
from sae_defense.sae import TopKSAE
from sae_defense.select import build_selected_features
from sae_defense.scoring import score_features
from sae_defense.utils import ensure_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--selected", default=None)
    parser.add_argument("--filter", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-gap", type=float, default=None)
    parser.add_argument("--max-benign-mean", type=float, default=None)
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def load_model_and_tokenizer(cfg: dict[str, Any]) -> tuple[Any, Any]:
    model_cfg = cfg["model"]
    dtype = torch_dtype(cfg["dtype"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["model_name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        dtype=dtype,
        attn_implementation=model_cfg.get("attn_implementation", "eager"),
        device_map=None,
    )
    model.to(cfg["device"])
    model.eval()
    return model, tokenizer


def inspect_modules(cfg: dict[str, Any], pattern: str | None) -> None:
    model_cfg = cfg["model"]
    config = AutoConfig.from_pretrained(
        model_cfg["model_name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=model_cfg.get("trust_remote_code", True),
            attn_implementation=model_cfg.get("attn_implementation", "eager"),
        )
    for name, module in model.named_modules():
        if not name:
            continue
        if pattern and pattern not in name:
            continue
        print(f"{name}\t{module.__class__.__name__}")


@torch.no_grad()
def collect_activations(cfg: dict[str, Any]) -> None:
    set_seed(cfg["seed"])
    model, tokenizer = load_model_and_tokenizer(cfg)
    records = load_jsonl(
        cfg["data"]["dataset_jsonl"],
        max_samples=cfg["data"].get("max_samples"),
    )

    output_dir = Path(cfg["output_dir"])
    act_dir = ensure_dir(output_dir / "activations")
    hook_storage: list[torch.Tensor] = []
    recorder = ActivationRecorder(capture=cfg["hook"]["capture"], storage=hook_storage)
    module = resolve_module(model, cfg["hook"]["module_path"])
    handle = module.register_forward_hook(recorder.hook)

    all_activations: list[torch.Tensor] = []
    example_ranges: list[dict[str, Any]] = []
    processed_records: list[dict[str, Any]] = []
    cursor = 0
    batch_size = int(cfg["data"]["batch_size"])
    truncate_to = int(cfg["data"]["truncate_to"])
    text_field = cfg["data"]["text_field"]
    id_field = cfg["data"]["id_field"]
    category_field = cfg["data"]["category_field"]

    for start in tqdm(range(0, len(records), batch_size), desc="collect"):
        batch = records[start : start + batch_size]
        texts = [item[text_field] for item in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=truncate_to,
            return_tensors="pt",
        )
        encoded = {key: value.to(cfg["device"]) for key, value in encoded.items()}
        hook_storage.clear()
        model(**encoded)
        if not hook_storage:
            raise RuntimeError("No activation captured. Check hook.module_path and hook.capture.")

        captured = hook_storage[-1]
        if captured.ndim != 3:
            raise RuntimeError(f"Expected [batch, seq, hidden] activation, got {tuple(captured.shape)}")

        attention_mask = encoded["attention_mask"].detach().cpu()
        for idx, item in enumerate(batch):
            valid_len = int(attention_mask[idx].sum().item())
            sample_acts = captured[idx, :valid_len, :].contiguous()
            all_activations.append(sample_acts)
            next_cursor = cursor + sample_acts.shape[0]
            example_ranges.append(
                {
                    "id": item[id_field],
                    "category": item[category_field],
                    "start": cursor,
                    "end": next_cursor,
                }
            )
            processed_records.append(item)
            cursor = next_cursor

    handle.remove()

    activations = torch.cat(all_activations, dim=0)
    payload = {
        "activations": activations,
        "example_ranges": example_ranges,
        "records": processed_records,
        "hidden_size": int(activations.shape[-1]),
        "num_tokens": int(activations.shape[0]),
        "hook": cfg["hook"],
    }
    torch.save(payload, act_dir / "activations.pt")


def train_sae(cfg: dict[str, Any]) -> None:
    set_seed(cfg["seed"])
    output_dir = Path(cfg["output_dir"])
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    act_payload = torch.load(output_dir / "activations" / "activations.pt", map_location="cpu")
    activations = act_payload["activations"].float()
    hidden_size = int(act_payload["hidden_size"])

    sae_cfg = cfg["sae"]
    model = TopKSAE(
        input_dim=hidden_size,
        width=int(sae_cfg["width"]),
        k=int(sae_cfg["k"]),
    ).to(cfg["device"])

    dataset = TensorDataset(activations)
    loader = DataLoader(
        dataset,
        batch_size=int(sae_cfg["batch_size_tokens"]),
        shuffle=True,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(sae_cfg["learning_rate"]))
    epochs = int(sae_cfg["epochs"])
    log_every = int(sae_cfg["log_every"])
    reconstruction_weight = float(sae_cfg["reconstruction_weight"])
    l1_weight = float(sae_cfg["l1_weight"])

    step = 0
    for epoch in range(epochs):
        progress = tqdm(loader, desc=f"train epoch {epoch + 1}/{epochs}")
        for (batch,) in progress:
            batch = batch.to(cfg["device"])
            recon, latent = model(batch)
            recon_loss = F.mse_loss(recon, batch)
            sparse_loss = latent.abs().mean()
            loss = reconstruction_weight * recon_loss + l1_weight * sparse_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if sae_cfg.get("normalize_decoder", True):
                model.normalize_decoder_()
            if step % log_every == 0:
                progress.set_postfix(
                    {
                        "loss": f"{loss.item():.6f}",
                        "recon": f"{recon_loss.item():.6f}",
                        "l0": f"{(latent > 0).float().sum(dim=-1).mean().item():.2f}",
                    }
                )
            step += 1

    checkpoint = {
        "state_dict": model.state_dict(),
        "hidden_size": hidden_size,
        "width": int(sae_cfg["width"]),
        "k": int(sae_cfg["k"]),
        "config": cfg,
    }
    torch.save(checkpoint, ckpt_dir / "sae_last.pt")


@torch.no_grad()
def score_sae_features(cfg: dict[str, Any]) -> None:
    output_dir = Path(cfg["output_dir"])
    act_payload = torch.load(output_dir / "activations" / "activations.pt", map_location="cpu")
    checkpoint = torch.load(output_dir / "checkpoints" / "sae_last.pt", map_location="cpu")

    model = TopKSAE(
        input_dim=int(checkpoint["hidden_size"]),
        width=int(checkpoint["width"]),
        k=int(checkpoint["k"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    activations = act_payload["activations"].float()
    ranges = act_payload["example_ranges"]
    labels = [item["category"] for item in ranges]

    example_latents: list[torch.Tensor] = []
    for item in tqdm(ranges, desc="score"):
        sample = activations[item["start"] : item["end"]]
        latent = model.encode(sample)
        pooled = latent.mean(dim=0)
        example_latents.append(pooled)

    feature_activations = torch.stack(example_latents, dim=0)
    scores = score_features(
        feature_activations=feature_activations,
        labels=labels,
        categories=sorted({item["category"] for item in ranges}),
    )
    save_json(output_dir / "feature_scores.json", scores)


@torch.no_grad()
def build_artifact(cfg: dict[str, Any], selected_path: str) -> None:
    output_dir = Path(cfg["output_dir"])
    checkpoint = torch.load(output_dir / "checkpoints" / "sae_last.pt", map_location="cpu")
    with Path(selected_path).open("r", encoding="utf-8") as handle:
        selected = json.load(handle)["features"]

    width = int(checkpoint["width"])
    hidden_size = int(checkpoint["hidden_size"])
    model = TopKSAE(
        input_dim=hidden_size,
        width=width,
        k=int(checkpoint["k"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    effective_delta_latent = build_effective_delta(width=width, selected=selected)
    decoder_weight = model.decoder.weight.detach().float()
    delta_dense = decoder_weight @ effective_delta_latent

    artifact_cfg = cfg["artifact"]
    payload = {
        "delta_dense": delta_dense.contiguous(),
        "effective_delta_latent": effective_delta_latent.contiguous(),
        "target_layer": int(cfg["hook"]["target_layer"]),
        "activation_site": cfg["hook"]["activation_site"],
        "alpha_default": float(artifact_cfg["alpha_default"]),
        "gamma": float(artifact_cfg["gamma"]),
        "application_direction": artifact_cfg["application_direction"],
        "model_id": artifact_cfg["model_id"],
        "sae_id": artifact_cfg["sae_id"],
        "artifact_version": artifact_cfg["artifact_version"],
        "tp_size": int(artifact_cfg["tp_size"]),
        "tp_rank": int(artifact_cfg["tp_rank"]),
        "delta_dense_layout": artifact_cfg["delta_dense_layout"],
    }
    payload["checksum"] = payload_checksum(payload)
    save_artifact(output_dir / "artifacts" / "sae_delta_artifact.pt", payload)


def select_features(
    cfg: dict[str, Any],
    *,
    top_k: int | None,
    min_gap: float | None,
    max_benign_mean: float | None,
) -> None:
    output_dir = Path(cfg["output_dir"])
    selection_cfg = cfg.get("selection", {})
    with (output_dir / "feature_scores.json").open("r", encoding="utf-8") as handle:
        scores = json.load(handle)

    payload = build_selected_features(
        scores=scores,
        top_k=int(top_k if top_k is not None else selection_cfg.get("top_k", 32)),
        min_gap=float(min_gap if min_gap is not None else selection_cfg.get("min_gap", 0.0)),
        max_benign_mean=(
            max_benign_mean if max_benign_mean is not None else selection_cfg.get("max_benign_mean")
        ),
        positive_only=bool(selection_cfg.get("positive_only", True)),
        weight_mode=str(selection_cfg.get("weight_mode", "gap")),
        normalize_weights=bool(selection_cfg.get("normalize_weights", True)),
    )
    save_json(output_dir / "selected_features.json", payload)


def main_collect() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    collect_activations(cfg)


def main_train() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_sae(cfg)


def main_score() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    score_sae_features(cfg)


def main_build_artifact() -> None:
    args = parse_args()
    if args.selected is None:
        raise ValueError("--selected is required for artifact build")
    cfg = load_config(args.config)
    build_artifact(cfg, args.selected)


def main_inspect_modules() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    inspect_modules(cfg, args.filter)


def main_select_features() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    select_features(
        cfg,
        top_k=args.top_k,
        min_gap=args.min_gap,
        max_benign_mean=args.max_benign_mean,
    )
