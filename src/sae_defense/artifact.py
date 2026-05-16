from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def build_effective_delta(width: int, selected: list[dict[str, Any]]) -> torch.Tensor:
    delta = torch.zeros(width, dtype=torch.float32)
    for item in selected:
        feature_id = int(item["id"])
        weight = float(item.get("weight", 1.0))
        delta[feature_id] = weight
    return delta


def payload_checksum(payload: dict[str, Any]) -> str:
    serializable = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            serializable[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sum": float(value.float().sum().item()),
            }
        else:
            serializable[key] = value
    encoded = json.dumps(serializable, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_artifact(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
