import torch
import triton
from torch import Tensor, nn
from ..kernel.rope import rope_kernel


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)

    Encodes position information into query and key vectors through rotation.
    Qwen2/LLaMA use a split-half layout that pairs the first and second halves
    of head_dim, allowing qᵀk attention scores to express relative positions.

    Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
               https://arxiv.org/abs/2104.09864
    """

    def __init__(
        self,
        head_dim: int,
        context_length: int,
        theta: float = 1_000_000.0,  # Qwen2.5 default; RoPE paper uses 10000.0
    ):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for split-half rotation"

        # Precompute frequencies: θᵢ = theta^{-2i/d}, i = 0, 1, ..., d/2 - 1
        # Shape: (head_dim // 2,)
        i = torch.arange(0, head_dim, 2, dtype=torch.float32)
        freqs = 1.0 / (theta ** (i / head_dim))

        # Precompute pos * θᵢ for all positions
        # Outer product shape: (context_length, head_dim // 2)
        positions = torch.arange(context_length, dtype=torch.float32)
        self.register_buffer("freqs", torch.outer(positions, freqs), persistent=False)

    def forward(self, x: Tensor, position_index: Tensor) -> Tensor:
        """Apply rotary position embedding to the input tensor.

        Args:
            x: Shape (heads, total_tokens_num, head_dim), with batches flattened.
            position_index: Shape (total_tokens_num), storing each token's
                position inside its request.

        Returns:
            Rotated tensor with the same shape.
        """
        head_num, token_num, head_dim = x.shape

        assert x.ndim == 3
        assert x.stride(-1) == 1
        assert head_dim == self.freqs.shape[1] * 2

        half_head_dim = head_dim // 2
        out = torch.empty_like(x)
        rope_kernel[(head_num, token_num)](
            x,
            self.freqs,
            position_index,
            out,
            x.stride(0),
            x.stride(1),
            out.stride(0),
            out.stride(1),
            half_head_dim,
            triton.next_power_of_2(half_head_dim),
        )

        return out
