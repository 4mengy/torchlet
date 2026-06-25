import torch
from torch import Tensor, nn

from .ffn_swiglu import FFNSwiGLU
from .gqa import GroupedQueryAttention
from .rms_norm import RmsNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        max_position_embeddings: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_bias: bool,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1_000_000.0,
    ):
        super().__init__()
        self.self_attn = GroupedQueryAttention(
            hidden_size,
            hidden_size,
            max_position_embeddings,
            num_attention_heads,
            num_key_value_heads,
            attention_bias,
            rope_theta,
        )
        self.mlp = FFNSwiGLU(hidden_size, intermediate_size)
        self.input_layernorm = RmsNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RmsNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
