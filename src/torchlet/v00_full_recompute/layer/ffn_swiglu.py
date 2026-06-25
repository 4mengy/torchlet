import torch.nn.functional as F
from torch import Tensor, nn


class FFNSwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # Elementwise multiplication acts as the gate
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
