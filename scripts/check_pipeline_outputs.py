from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse

import torch

from sae_defense.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"])

    activations_path = output_dir / "activations" / "activations.pt"
    checkpoint_path = output_dir / "checkpoints" / "sae_last.pt"
    scores_path = output_dir / "feature_scores.json"
    selected_path = output_dir / "selected_features.json"
    artifact_path = output_dir / "artifacts" / "sae_delta_artifact.pt"
    full_artifact_path = output_dir / "artifacts" / "sae_full_artifact.pt"

    print("=== File Check ===")
    for path in [activations_path, checkpoint_path, scores_path, selected_path, artifact_path, full_artifact_path]:
        print(f"{path}: {'OK' if path.exists() else 'MISSING'}")

    if not activations_path.exists() or not checkpoint_path.exists() or not artifact_path.exists():
        print("\nMissing required files. Stop here.")
        return

    activations = torch.load(activations_path, map_location="cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    artifact = torch.load(artifact_path, map_location="cpu")
    full_artifact = torch.load(full_artifact_path, map_location="cpu") if full_artifact_path.exists() else None

    act_tensor = activations["activations"]
    hidden_size = int(activations["hidden_size"])
    num_examples = len(activations["example_ranges"])
    width = int(checkpoint["width"])
    k = int(checkpoint["k"])
    delta_dense = artifact["delta_dense"]
    effective_delta_latent = artifact["effective_delta_latent"]

    print("\n=== Shape Check ===")
    print(f"activations.shape: {tuple(act_tensor.shape)}")
    print(f"hidden_size: {hidden_size}")
    print(f"num_examples: {num_examples}")
    print(f"sae_width: {width}")
    print(f"sae_k: {k}")
    print(f"delta_dense.shape: {tuple(delta_dense.shape)}")
    print(f"effective_delta_latent.shape: {tuple(effective_delta_latent.shape)}")
    if full_artifact is not None:
        print(f"full.encoder_weight.shape: {tuple(full_artifact['encoder_weight'].shape)}")
        print(f"full.decoder_weight.shape: {tuple(full_artifact['decoder_weight'].shape)}")
        print(f"full.input_bias.shape: {tuple(full_artifact['input_bias'].shape)}")

    print("\n=== Value Check ===")
    print(f"delta_dense has nan: {bool(torch.isnan(delta_dense).any().item())}")
    print(f"delta_dense has inf: {bool(torch.isinf(delta_dense).any().item())}")
    print(f"latent has nan: {bool(torch.isnan(effective_delta_latent).any().item())}")
    print(f"latent has inf: {bool(torch.isinf(effective_delta_latent).any().item())}")

    print("\n=== Metadata Check ===")
    print(f"target_layer: {artifact.get('target_layer')}")
    print(f"activation_site: {artifact.get('activation_site')}")
    print(f"alpha_default: {artifact.get('alpha_default')}")
    print(f"gamma: {artifact.get('gamma')}")
    print(f"checksum exists: {'checksum' in artifact}")
    if full_artifact is not None:
        print(f"full artifact checksum exists: {'checksum' in full_artifact}")

    print("\n=== Simple Verdict ===")
    ok = True
    if act_tensor.ndim != 2:
        print("- activations tensor is not 2D")
        ok = False
    if act_tensor.shape[1] != hidden_size:
        print("- hidden_size does not match activations.shape[1]")
        ok = False
    if delta_dense.ndim != 1 or delta_dense.shape[0] != hidden_size:
        print("- delta_dense shape does not match hidden_size")
        ok = False
    if effective_delta_latent.ndim != 1 or effective_delta_latent.shape[0] != width:
        print("- effective_delta_latent shape does not match SAE width")
        ok = False
    if torch.isnan(delta_dense).any() or torch.isinf(delta_dense).any():
        print("- delta_dense contains nan/inf")
        ok = False
    if torch.isnan(effective_delta_latent).any() or torch.isinf(effective_delta_latent).any():
        print("- effective_delta_latent contains nan/inf")
        ok = False
    if full_artifact is not None:
        if full_artifact["encoder_weight"].shape != (width, hidden_size):
            print("- full artifact encoder_weight shape does not match [width, hidden_size]")
            ok = False
        if full_artifact["decoder_weight"].shape != (hidden_size, width):
            print("- full artifact decoder_weight shape does not match [hidden_size, width]")
            ok = False
        if full_artifact["input_bias"].shape != (hidden_size,):
            print("- full artifact input_bias shape does not match [hidden_size]")
            ok = False
        if int(full_artifact["k"]) != k:
            print("- full artifact k does not match checkpoint k")
            ok = False

    if ok:
        print("PASS: pipeline outputs are structurally valid.")
    else:
        print("FAIL: one or more structural checks failed.")


if __name__ == "__main__":
    main()
