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

    def forward(self, x: Tensor) -> Tensor:
        """Apply rotary position embedding to the input tensor.

        Args:
            x: Shape (batch, heads, num_tokens, head_dim).
               Usually Q or K after transpose(1, 2).

        Returns:
            Rotated tensor with the same shape.
        """
        *_, num_tokens, _ = x.shape

        # Select frequencies for the current sequence length
        # Shape: (num_tokens, head_dim // 2)
        freqs = self.freqs[:num_tokens, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype, device=x.device)
        sin = emb.sin().to(dtype=x.dtype, device=x.device)

        # Expand dimensions for broadcasting:
        # cos/sin: (num_tokens, head_dim) -> (1, 1, num_tokens, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)  # type: ignore[attr-defined]
        sin = sin.unsqueeze(0).unsqueeze(0)  # type: ignore[attr-defined]

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


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    num_heads = 4
    num_tokens = 8
    head_dim = 64

    rope = RotaryPositionEmbedding(
        head_dim=head_dim,
        context_length=num_tokens * 3,  # Precompute a longer sequence
        theta=10000.0,
    )

    x = torch.randn(batch_size, num_heads, num_tokens, head_dim)
    y = rope(x)

    # Rotation does not change the shape
    assert y.shape == x.shape, f"Output shape {y.shape} does not match input {x.shape}"

    # Rotation preserves each vector's L2 norm because it is an orthogonal transform
    x_norm = torch.norm(x.float(), dim=-1)
    y_norm = torch.norm(y.float(), dim=-1)
    assert torch.allclose(x_norm, y_norm, atol=1e-5), "RoPE should preserve L2 norm"

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Max norm difference: {(x_norm - y_norm).abs().max().item():.2e}")
    print("All assertions passed!")
