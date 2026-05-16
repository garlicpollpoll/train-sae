from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


def resolve_module(root: nn.Module, module_path: str) -> nn.Module:
    current: Any = root
    for part in module_path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    if not isinstance(current, nn.Module):
        raise TypeError(f"Resolved object is not a module: {module_path}")
    return current


def extract_tensor(obj: Any) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, tuple):
        for item in obj:
            if isinstance(item, torch.Tensor):
                return item
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError("Could not extract tensor from hooked object")


@dataclass
class ActivationRecorder:
    capture: str
    storage: list[torch.Tensor]

    def hook(self, module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
        if self.capture == "input":
            tensor = extract_tensor(args[0] if len(args) == 1 else args)
        elif self.capture == "output":
            tensor = extract_tensor(output)
        else:
            raise ValueError(f"Unsupported capture mode: {self.capture}")
        self.storage.append(tensor.detach().to("cpu", dtype=torch.float32))
