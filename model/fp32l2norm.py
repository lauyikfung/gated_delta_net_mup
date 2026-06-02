from __future__ import annotations

import torch
import torch.nn as nn


class FP32L2Norm(nn.Module):
    def __init__(self, *, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_float = x.float()
        inv_norm = torch.rsqrt(torch.sum(x_float * x_float, dim=self.dim, keepdim=True) + self.eps)
        return (x_float * inv_norm).to(dtype=orig_dtype)
