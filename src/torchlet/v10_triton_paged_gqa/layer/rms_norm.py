import torch
import triton
from torch import Tensor, nn
from ..kernel.rms_norm import rms_norm_kernel


class RmsNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps  # Avoid a zero denominator
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        assert x.dim() == 2
        assert x.is_cuda

        x = x.contiguous()
        B, H = x.shape

        assert H == self.weight.numel()

        out = torch.empty_like(x)
        H_PAD = triton.next_power_of_2(H)
        rms_norm_kernel[(B,)](x, self.weight, self.eps, out, H, H_PAD)
        return out
