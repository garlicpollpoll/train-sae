from __future__ import annotations

from typing import Any

import torch


def score_features(
    feature_activations: torch.Tensor,
    labels: list[str],
    categories: list[str],
) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    label_to_mask = {
        category: torch.tensor([label == category for label in labels], dtype=torch.bool)
        for category in categories
    }

    benign_mask = label_to_mask.get("benign")
    jailbreak_mask = label_to_mask.get("jailbreak")
    if benign_mask is None or jailbreak_mask is None:
        raise ValueError("Both benign and jailbreak categories must exist for scoring")

    benign_mean = feature_activations[benign_mask].mean(dim=0)
    jailbreak_mean = feature_activations[jailbreak_mask].mean(dim=0)
    gaps = jailbreak_mean - benign_mean

    for feature_id, gap in enumerate(gaps.tolist()):
        scores.append(
            {
                "feature_id": feature_id,
                "jailbreak_mean": float(jailbreak_mean[feature_id].item()),
                "benign_mean": float(benign_mean[feature_id].item()),
                "gap": float(gap),
            }
        )
    scores.sort(key=lambda item: item["gap"], reverse=True)
    return scores
