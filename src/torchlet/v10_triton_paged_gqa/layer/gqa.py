import torch
import triton
from torch import Tensor, nn

from ..forward_params import ForwardParams
from ..kernel.gqa import (
    forward_ragged_attention_kernel,
    forward_slot_attention_decode_kernel,
)
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

        # Decode always uses fixed slot-shaped tensors for CUDA Graph replay.
        # queries: (num_heads, max_decode_slots, head_dim)
        # keys/values: (num_kv_heads, max_decode_slots, head_dim)
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
            forward_params.block_table,
        )

        # tensor shape: (total_tokens_num, d_out)
        context_vec = self._forward_ragged_attention_triton(queries, forward_params)
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
            forward_params.block_table,
        )

        # tensor shape: (max_decode_slots, d_out)
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

    def _forward_ragged_attention_triton(
        self,
        queries: Tensor,
        forward_params: ForwardParams,
    ) -> Tensor:
        """Run causal prefill attention over the paged KV cache."""
        query_block = 16
        req_indptr = forward_params.req_indptr_cpu
        slot_ids = forward_params.slot_ids_cpu
        assert slot_ids is not None
        assert len(slot_ids) + 1 == len(req_indptr)

        if queries.shape[1] == 0:
            return queries.new_empty((0, self.d_out))

        tiles = forward_params.prefill_attention_tiles
        if tiles is None:
            request_start = req_indptr[:-1]
            request_len = req_indptr[1:] - request_start
            tiles_per_request = torch.div(
                request_len + query_block - 1,
                query_block,
                rounding_mode="floor",
            )
            total_tiles = int(tiles_per_request.sum())

            # Default to full tiles, then fix the last tile of each request.
            tile_query_len = torch.full(
                (total_tiles,),
                query_block,
                dtype=req_indptr.dtype,
                device=req_indptr.device,
            )
            nonempty_request = tiles_per_request > 0
            last_tile_idx = torch.cumsum(tiles_per_request, dim=0)[nonempty_request] - 1
            tile_query_len[last_tile_idx] = (
                request_len[nonempty_request]
                - (tiles_per_request[nonempty_request] - 1) * query_block
            )

            # The exclusive prefix sum gives absolute starts in the flat
            # ragged token buffer.
            tile_query_start = torch.cat(
                (
                    req_indptr.new_zeros(1),
                    torch.cumsum(tile_query_len, dim=0)[:-1],
                )
            )
            tile_request_idx = torch.repeat_interleave(
                torch.arange(request_len.numel(), device=req_indptr.device),
                tiles_per_request,
            )
            tile_request_start = request_start[tile_request_idx]
            tile_slot = torch.as_tensor(
                slot_ids,
                dtype=req_indptr.dtype,
                device=req_indptr.device,
            )[tile_request_idx]

            def device_int_tensor(values):
                return values.to(dtype=torch.int32, device=queries.device)

            tiles = (
                device_int_tensor(tile_query_start),
                device_int_tensor(tile_request_start),
                device_int_tensor(tile_query_len),
                device_int_tensor(tile_slot),
            )
            forward_params.prefill_attention_tiles = tiles

        (
            tile_query_start_gpu,
            tile_request_start_gpu,
            tile_query_len_gpu,
            tile_slot_gpu,
        ) = tiles

        kvcache = forward_params.kvcache
        k_layer = kvcache.cache[0, self.layer_idx]
        v_layer = kvcache.cache[1, self.layer_idx]
        total_tokens = queries.shape[1]
        context_vec = queries.new_empty(total_tokens, self.num_heads, self.head_dim)

        assert queries.stride(-1) == 1
        assert k_layer.stride(-1) == 1
        assert k_layer.stride() == v_layer.stride()
        assert context_vec.is_contiguous()
        assert forward_params.block_table.stride(1) == 1

        grid = (tile_query_start_gpu.numel(), self.num_heads)
        forward_ragged_attention_kernel[grid](
            tile_query_start_gpu,
            tile_request_start_gpu,
            tile_query_len_gpu,
            tile_slot_gpu,
            queries,
            queries.stride(0),
            queries.stride(1),
            queries.stride(2),
            forward_params.block_table,
            forward_params.block_table.stride(0),
            k_layer,
            v_layer,
            k_layer.stride(0),
            k_layer.stride(1),
            k_layer.stride(2),
            context_vec,
            context_vec.stride(0),
            context_vec.stride(1),
            BLOCK_SIZE=kvcache.block_size,
            BLOCK_SIZE_PAD=max(16, triton.next_power_of_2(kvcache.block_size)),
            KV_GROUPS=self.num_heads // self.num_kv_heads,
            QUERY_BLOCK=query_block,
            HEAD_DIM=self.head_dim,
            HEAD_DIM_PAD=max(16, triton.next_power_of_2(self.head_dim)),
            SM_SCALE=self.head_dim**-0.5,
            num_warps=4,
        )
        return context_vec.view(total_tokens, self.d_out)

    def _forward_slot_attention_decode(
        self,
        queries: Tensor,
        forward_params: ForwardParams,
    ) -> Tensor:
        """
        Run one-token decode attention over fixed decode slots.

        Inactive slots use dummy input tokens and are zeroed by active_mask.
        queries: (num_heads, max_decode_slots, head_dim)
        """
        num_kv_groups = self.num_heads // self.num_kv_heads
        max_slots = queries.shape[1]

        # Group query heads by the KV head they attend to:
        # (num_heads, max_decode_slots, head_dim)
        # -> (num_kv_heads, num_kv_groups, max_decode_slots, head_dim)
        queries = queries.view(
            self.num_kv_heads, num_kv_groups, max_slots, self.head_dim
        )

        kvcache = forward_params.kvcache
        k_layer = kvcache.cache[0, self.layer_idx]
        v_layer = kvcache.cache[1, self.layer_idx]
        context_vec = queries.new_zeros(max_slots, self.d_out)

        assert queries.stride(-1) == 1
        assert k_layer.stride(-1) == 1
        assert k_layer.stride() == v_layer.stride()
        assert context_vec.is_contiguous()
        assert forward_params.position_index.shape == (max_slots,)
        assert forward_params.position_index.stride(0) == 1
        assert forward_params.active_mask.shape == (max_slots,)
        assert forward_params.active_mask.stride(0) == 1
        assert forward_params.block_table.shape[0] == max_slots
        assert forward_params.block_table.stride(1) == 1

        grid = (max_slots, self.num_kv_heads)
        forward_slot_attention_decode_kernel[grid](
            forward_params.active_mask,
            forward_params.position_index,
            queries,
            queries.stride(0),
            queries.stride(1),
            queries.stride(2),
            queries.stride(3),
            forward_params.block_table,
            forward_params.block_table.stride(0),
            k_layer,
            v_layer,
            k_layer.stride(0),
            k_layer.stride(1),
            k_layer.stride(2),
            context_vec,
            KV_HEAD_NUM=self.num_kv_heads,
            BLOCK_SIZE=kvcache.block_size,
            BLOCK_SIZE_PAD=max(16, triton.next_power_of_2(kvcache.block_size)),
            KV_GROUPS=num_kv_groups,
            HEAD_DIM=self.head_dim,
            KV_GROUPS_PAD=max(16, triton.next_power_of_2(num_kv_groups)),
            HEAD_DIM_PAD=max(16, triton.next_power_of_2(self.head_dim)),
            SM_SCALE=self.head_dim**-0.5,
            num_warps=4,
        )
        return context_vec
