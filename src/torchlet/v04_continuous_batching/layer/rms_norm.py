import torch
from torch import Tensor, nn


class RmsNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps  # Avoid a zero denominator
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        assert x.dim() == 2
        norm = x * torch.rsqrt(torch.pow(x, 2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight  # Elementwise scale along the last dimension
