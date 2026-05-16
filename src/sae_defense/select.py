from __future__ import annotations

from typing import Any


def build_selected_features(
    scores: list[dict[str, Any]],
    *,
    top_k: int,
    min_gap: float,
    max_benign_mean: float | None,
    positive_only: bool,
    weight_mode: str,
    normalize_weights: bool,
) -> dict[str, Any]:
    filtered: list[dict[str, Any]] = []
    for item in scores:
        gap = float(item["gap"])
        benign_mean = float(item["benign_mean"])
        if positive_only and gap <= 0:
            continue
        if gap < min_gap:
            continue
        if max_benign_mean is not None and benign_mean > max_benign_mean:
            continue
        filtered.append(item)

    selected = filtered[:top_k]
    if not selected:
        raise ValueError("No features matched the selection criteria")

    raw_weights: list[float] = []
    for item in selected:
        if weight_mode == "gap":
            raw_weights.append(max(float(item["gap"]), 0.0))
        elif weight_mode == "unit":
            raw_weights.append(1.0)
        else:
            raise ValueError(f"Unsupported weight_mode: {weight_mode}")

    if normalize_weights:
        max_weight = max(raw_weights)
        denom = max_weight if max_weight > 0 else 1.0
        weights = [weight / denom for weight in raw_weights]
    else:
        weights = raw_weights

    features: list[dict[str, float | int]] = []
    for item, weight in zip(selected, weights):
        features.append(
            {
                "id": int(item["feature_id"]),
                "weight": float(weight),
                "gap": float(item["gap"]),
                "benign_mean": float(item["benign_mean"]),
                "jailbreak_mean": float(item["jailbreak_mean"]),
            }
        )

    return {
        "features": features,
        "selection_policy": {
            "top_k": int(top_k),
            "min_gap": float(min_gap),
            "max_benign_mean": None if max_benign_mean is None else float(max_benign_mean),
            "positive_only": bool(positive_only),
            "weight_mode": weight_mode,
            "normalize_weights": bool(normalize_weights),
        },
        "num_candidates_before_filter": len(scores),
        "num_selected": len(features),
    }
