import torch
from torch import Tensor, nn


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

        self.head_dim = head_dim
        self.context_length = context_length
        self.theta = theta

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
        freqs = self.freqs[position_index, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype, device=x.device)
        sin = emb.sin().to(dtype=x.dtype, device=x.device)

        x_rope = (x * cos) + (self._rotate_half(x) * sin)

        return x_rope

    def _rotate_half(self, x: Tensor) -> Tensor:
        """Rotate hidden dimensions with Qwen2's split-half convention.

        For vector [x₀, x₁, x₂, x₃, x₄, x₅, x₆, x₇]:
        1. Split it into two halves:
           x1 = [x₀, x₁, x₂, x₃]
           x2 = [x₄, x₅, x₆, x₇]
        2. Concatenate [-x₄, -x₅, -x₆, -x₇, x₀, x₁, x₂, x₃]
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
