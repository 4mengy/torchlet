import torch
import triton
from torch import Tensor, nn
from ..kernel.ffn_swiglu import down_kernel, swiglu_kernel


class FFNSwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 2
        assert x.is_cuda

        x = x.contiguous()
        M, H = x.shape
        D = self.gate_proj.out_features

        z = torch.empty((M, D), device=x.device, dtype=x.dtype)

        # Kernel 1：
        # z = silu(x @ wg.T) * (x @ wu.T)
        def swiglu_grid(meta):
            return (
                triton.cdiv(M, meta["BLOCK_M"]),
                triton.cdiv(D, meta["BLOCK_D"]),
            )

        swiglu_kernel[swiglu_grid](
            x,
            self.gate_proj.weight,
            self.up_proj.weight,
            z,
            M,
            H,
            D,
        )

        y = torch.empty((M, H), device=x.device, dtype=x.dtype)

        # Kernel 2：
        # y = z @ wd.T
        def down_grid(meta):
            return (
                triton.cdiv(M, meta["BLOCK_M"]),
                triton.cdiv(H, meta["BLOCK_H"]),
            )

        down_kernel[down_grid](
            z,
            self.down_proj.weight,
            y,
            M,
            D,
            H,
        )
        return y
