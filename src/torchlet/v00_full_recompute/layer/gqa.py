import torch
from torch import Tensor, nn

from .rope import RotaryPositionEmbedding


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        num_heads: int,
        num_kv_heads: int,
        qkv_bias: bool = False,
        rope_theta: float = 1_000_000.0,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert 0 < num_kv_heads <= num_heads, (
            "num_kv_heads must be less than or equal to num_heads and greater than 0"
        )
        assert num_heads % num_kv_heads == 0, (
            "num_heads must be divisible by num_kv_heads"
        )

        self.d_out = d_out
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_out // num_heads  # Dimension of each query head
        # Queries use num_heads; keys and values use num_kv_heads.
        kv_head_dim = self.head_dim * num_kv_heads
        self.q_proj = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.k_proj = nn.Linear(d_in, kv_head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(d_in, kv_head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(
            d_out, d_out, bias=False
        )  # Linear projection that combines head outputs
        self.pos_emb = RotaryPositionEmbedding(
            self.head_dim, context_length, rope_theta
        )

    def forward(self, x: Tensor) -> Tensor:
        b, num_tokens, _ = x.shape
        queries = self.q_proj(x)  # Tensor shape: (b, num_tokens, d_out)
        keys = self.k_proj(x)  # Tensor shape: (b, num_tokens, kv_head_dim)
        values = self.v_proj(x)

        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_kv_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_kv_heads, self.head_dim)

        # From shape (b, num_tokens, num_heads, head_dim)
        # to (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        queries = self.pos_emb(queries)
        keys = self.pos_emb(keys)

        # repeat kv heads to match num_heads
        num_kv_groups = self.num_heads // self.num_kv_heads
        keys = keys.repeat_interleave(num_kv_groups, dim=1)
        values = values.repeat_interleave(num_kv_groups, dim=1)

        # Dot product per head: (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = torch.triu(
            torch.ones(
                num_tokens,
                num_tokens,
                dtype=torch.bool,
                device=attn_scores.device,
            ),
            diagonal=1,
        )

        # Mask future positions in the attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # Tensor shape: (b, num_tokens, n_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.o_proj(context_vec)  # Output linear projection
        return context_vec
