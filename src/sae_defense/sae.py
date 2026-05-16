from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    def __init__(self, input_dim: int, width: int, k: int):
        super().__init__()
        self.input_dim = input_dim
        self.width = width
        self.k = k
        self.encoder = nn.Linear(input_dim, width, bias=True)
        self.decoder = nn.Linear(width, input_dim, bias=False)
        self.input_bias = nn.Parameter(torch.zeros(input_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - self.input_bias
        pre = self.encoder(centered)
        act = F.relu(pre)
        values, indices = torch.topk(act, k=min(self.k, act.shape[-1]), dim=-1)
        sparse = torch.zeros_like(act)
        sparse.scatter_(dim=-1, index=indices, src=values)
        return sparse

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) + self.input_bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.decoder.weight.div_(norms)
