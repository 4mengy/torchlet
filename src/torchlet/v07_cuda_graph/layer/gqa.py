import torch
from torch import Tensor, nn

from ..forward_params import ForwardParams
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
        layer_idx: int = 0,
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
        self.layer_idx = layer_idx

    def forward(self, x: Tensor, forward_params: ForwardParams) -> Tensor:
        queries, keys, values = self._project_qkv(x)
        # queries: (num_heads, seq_len, head_dim)
        # keys: (num_kv_heads, seq_len, head_dim)
        # values: (num_kv_heads, seq_len, head_dim)
        if forward_params.is_prefill:
            return self._forward_prefill(queries, keys, values, forward_params)

        # Batched decode carries one new token per active request.
        # queries: (num_heads, req_num, head_dim)
        # keys/values: (num_kv_heads, req_num, head_dim)
        return self._forward_decode(queries, keys, values, forward_params)

    def _forward_prefill(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        forward_params: ForwardParams,
    ) -> Tensor:
        # (num_heads, seq_len, head_dim), (num_kv_heads, seq_len, head_dim)
        queries, keys = self._apply_rope(queries, keys, forward_params.position_index)
        forward_params.kvcache.append_prefill(
            keys,
            values,
            self.layer_idx,
            forward_params.slot_ids_cpu,
            forward_params.req_indptr_cpu,
        )
        # tensor shape: (total_tokens_num, d_out)
        context_vec = self._forward_ragged_attention(
            queries,
            keys,
            values,
            forward_params,
        )
        return self.o_proj(context_vec)

    def _forward_decode(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        forward_params: ForwardParams,
    ) -> Tensor:
        # Apply RoPE to the new decode tokens before appending K to the cache.
        queries, keys = self._apply_rope(queries, keys, forward_params.position_index)

        forward_params.kvcache.append_decode(
            keys,
            values,
            self.layer_idx,
            forward_params.position_index,
            forward_params.active_mask,
        )

        # tensor shape: (req_num, d_out)
        context_vec = self._forward_slot_attention_decode(
            queries,
            forward_params,
        )
        return self.o_proj(context_vec)

    def _project_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project flat tokens into per-head Q/K/V tensors.

        Returns:
            queries: (num_heads, total_tokens_num, head_dim)
            keys: (num_kv_heads, total_tokens_num, head_dim)
            values: (num_kv_heads, total_tokens_num, head_dim)
        """
        total_tokens_num = x.shape[0]

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.view(total_tokens_num, self.num_heads, self.head_dim)
        keys = keys.view(total_tokens_num, self.num_kv_heads, self.head_dim)
        values = values.view(total_tokens_num, self.num_kv_heads, self.head_dim)

        queries = queries.transpose(0, 1)
        keys = keys.transpose(0, 1)
        values = values.transpose(0, 1)

        return queries, keys, values

    def _apply_rope(
        self,
        queries: Tensor,
        keys: Tensor,
        position_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # queries: (num_heads, total_tokens_num, head_dim)
        # keys: (num_kv_heads, total_tokens_num, head_dim)
        queries = self.pos_emb(queries, position_index)
        keys = self.pos_emb(keys, position_index)
        return queries, keys

    def _forward_ragged_attention(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        forward_params: ForwardParams,
    ) -> Tensor:
        """
        Run per-request causal attention over a flat ragged token buffer.

        queries: (num_heads, total_tokens_num, head_dim)
        keys/values: (num_kv_heads, total_tokens_num, head_dim)
        """
        total_tokens_num = queries.shape[1]
        num_kv_groups = self.num_heads // self.num_kv_heads
        # Group query heads by the KV head they attend to:
        # (num_heads, total_tokens_num, head_dim)
        # -> (num_kv_heads, num_kv_groups, total_tokens_num, head_dim)
        queries = queries.view(
            self.num_kv_heads, num_kv_groups, total_tokens_num, self.head_dim
        )

        req_indptr_cpu = forward_params.req_indptr_cpu
        req_len_cpu = req_indptr_cpu[1:] - req_indptr_cpu[:-1]
        req_indptr_list = req_indptr_cpu.tolist()
        req_len_list = req_len_cpu.tolist()
        context_vec_list = list()
        for i, req_len in enumerate(req_len_list):
            start = req_indptr_list[i]
            end = req_indptr_list[i + 1]

            # (num_kv_heads, num_kv_groups, num_tokens, head_dim)
            sub_q = queries[:, :, start:end, :]
            # (num_kv_heads, 1, num_tokens, head_dim)
            sub_k = keys[:, start:end, :].unsqueeze(1)
            sub_v = values[:, start:end, :].unsqueeze(1)

            # Dot product per head: (num_kv_heads, num_kv_groups, num_tokens, num_tokens)
            attn_scores = sub_q @ sub_k.transpose(2, 3)
            mask_bool = torch.triu(
                torch.ones(
                    req_len,
                    req_len,
                    dtype=torch.bool,
                    device=attn_scores.device,
                ),
                diagonal=1,
            )

            # Mask future positions in the attention scores
            attn_scores.masked_fill_(mask_bool, -torch.inf)

            attn_weights = torch.softmax(attn_scores / sub_k.shape[-1] ** 0.5, dim=-1)

            # Tensor shape: (num_tokens, n_heads, head_dim)
            req_context_vec = (
                (attn_weights @ sub_v)
                .view(self.num_heads, req_len, self.head_dim)
                .transpose(0, 1)
            )
            context_vec_list.append(req_context_vec)

        context_vec = torch.cat(context_vec_list, dim=0)
        return context_vec.contiguous().view(total_tokens_num, self.d_out)

    def _forward_slot_attention_decode(
        self,
        queries: Tensor,
        forward_params: ForwardParams,
    ) -> Tensor:
        """
        Run one-token decode attention against each request's cached K/V.

        queries: (num_heads, req_num, head_dim)
        """
        num_kv_groups = self.num_heads // self.num_kv_heads

        max_slots = queries.shape[1]

        # Group query heads by the KV head they attend to:
        # (num_heads, req_num, head_dim)
        # -> (num_kv_heads, num_kv_groups, req_num, head_dim)
        queries = queries.view(
            self.num_kv_heads, num_kv_groups, max_slots, self.head_dim
        )

        context_vec = queries.new_zeros(max_slots, self.d_out)
        for slot in range(max_slots):
            # (num_kv_heads, num_kv_groups, 1, head_dim)
            sub_q = queries[:, :, slot : slot + 1, :]

            # Cached K/V for this request:
            # (num_kv_heads, max_seq_len, head_dim)
            sub_k, sub_v = forward_params.kvcache.get_slot(self.layer_idx, slot)
            # (num_kv_heads, 1, max_seq_len, head_dim)
            sub_k = sub_k.unsqueeze(1)
            sub_v = sub_v.unsqueeze(1)

            # Dot product per head: (num_kv_heads, num_kv_groups, 1, max_seq_len)
            attn_scores = sub_q @ sub_k.transpose(2, 3)

            valid = (
                forward_params.kvcache.cache_positions
                <= forward_params.position_index[slot]
            )
            attn_scores = attn_scores.masked_fill(
                ~valid.view(1, 1, 1, forward_params.kvcache.max_seq_len),
                torch.finfo(attn_scores.dtype).min,
            )

            attn_weights = torch.softmax(attn_scores / sub_k.shape[-1] ** 0.5, dim=-1)

            # Tensor shape: (1, n_heads, head_dim)
            req_context_vec = (
                (attn_weights @ sub_v)
                .view(self.num_heads, 1, self.head_dim)
                .transpose(0, 1)
            )

            active = forward_params.active_mask[slot].to(req_context_vec.dtype)
            context_vec[slot] = req_context_vec.contiguous().view(self.d_out) * active
        return context_vec
