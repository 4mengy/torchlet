import triton
import triton.language as tl


@triton.jit
def forward_ragged_attention_kernel(
    tile_query_start,
    tile_request_start,
    tile_query_len,
    tile_slot,
    queries,  # (num_heads, total_tokens, head_dim)
    q_stride_head,
    q_stride_token,
    q_stride_dim,
    block_table,  # [max_decode_slots, max_blocks_per_req]
    block_table_stride_slot,
    k_layer,
    v_layer,
    cache_stride_block,
    cache_stride_kv_head,
    cache_stride_token,
    out,  # (total_tokens, num_heads, head_dim)
    out_stride_token,
    out_stride_head,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_PAD: tl.constexpr,
    KV_GROUPS: tl.constexpr,
    QUERY_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,
    SM_SCALE: tl.constexpr,
):
    tid = tl.program_id(0)
    hid = tl.program_id(1)
    kvhid = hid // KV_GROUPS

    query_start = tl.load(tile_query_start + tid)
    request_start = tl.load(tile_request_start + tid)
    query_len = tl.load(tile_query_len + tid)
    slot = tl.load(tile_slot + tid)

    offs_query = tl.arange(0, QUERY_BLOCK)
    mask_query = offs_query < query_len
    offs_dim = tl.arange(0, HEAD_DIM_PAD)
    mask_dim = offs_dim < HEAD_DIM

    # q: (query_block, head_dim_pad)
    query_offsets = query_start + offs_query
    q = tl.load(
        queries
        + hid * q_stride_head
        + query_offsets[:, None] * q_stride_token
        + offs_dim[None, :] * q_stride_dim,
        mask=mask_query[:, None] & mask_dim[None, :],
        other=0.0,
    )

    query_positions = query_offsets - request_start
    query_end = query_start - request_start + query_len
    num_blocks = tl.cdiv(query_end, BLOCK_SIZE)

    # Keep one online-softmax state per query in this tile. Invalid rows use
    # finite dummy scores so a short final tile never evaluates -inf - -inf.
    scores_max = tl.full((QUERY_BLOCK,), float("-inf"), tl.float32)
    weights_sum = tl.zeros((QUERY_BLOCK,), tl.float32)
    acc = tl.zeros((QUERY_BLOCK, HEAD_DIM_PAD), tl.float32)

    block_offsets = tl.arange(0, BLOCK_SIZE_PAD)
    block_mask = block_offsets < BLOCK_SIZE
    for logical_block_id in tl.range(0, num_blocks):
        key_positions = logical_block_id * BLOCK_SIZE + block_offsets

        physical_block_id = tl.load(
            block_table + slot * block_table_stride_slot + logical_block_id
        )

        cache_mask = block_mask[:, None] & mask_dim[None, :]
        # k_layer/v_layer: (block_id, num_kv_heads, block_size, head_dim)
        cache_offsets = (
            physical_block_id * cache_stride_block
            + kvhid * cache_stride_kv_head
            + block_offsets[:, None] * cache_stride_token
            + offs_dim[None, :]
        )

        # keys/values: (block_size_pad, head_dim_pad)
        k_value = tl.load(k_layer + cache_offsets, mask=cache_mask, other=0.0)
        v_value = tl.load(v_layer + cache_offsets, mask=cache_mask, other=0.0)

        # scores: (query_block, block_size_pad)
        scores = tl.dot(q, tl.trans(k_value)) * SM_SCALE
        causal_mask = (
            mask_query[:, None]
            & block_mask[None, :]
            & (key_positions[None, :] <= query_positions[:, None])
        )
        masked_score = tl.where(mask_query[:, None], float("-inf"), 0.0)
        scores = tl.where(causal_mask, scores, masked_score)

        tile_max = tl.max(scores, axis=1)
        new_scores_max = tl.maximum(scores_max, tile_max)
        correction = tl.exp(scores_max - new_scores_max)
        weights = tl.exp(scores - new_scores_max[:, None])
        weights = tl.where(causal_mask, weights, 0.0)

        acc = acc * correction[:, None]
        acc = tl.dot(weights.to(v_value.dtype), v_value, acc=acc)
        weights_sum = weights_sum * correction + tl.sum(weights, axis=1)
        scores_max = new_scores_max

    result = acc / weights_sum[:, None]
    result = tl.where(mask_query[:, None], result, 0.0)

    tl.store(
        out
        + query_offsets[:, None] * out_stride_token
        + hid * out_stride_head
        + offs_dim[None, :],
        result,
        mask=mask_query[:, None] & mask_dim[None, :],
    )


@triton.jit
def forward_slot_attention_decode_kernel(
    active_mask,
    position_index,
    queries,  # (num_kv_heads, num_kv_groups, slot, head_dim)
    q_stride_kv_h,
    q_stride_kv_g,
    q_stride_slot,
    q_stride_kv_d,
    block_table,  # [max_decode_slots, max_blocks_per_req]
    block_table_stride_slot,
    k_layer,
    v_layer,
    cache_stride_block,
    cache_stride_kv_head,
    cache_stride_token,
    out,
    KV_HEAD_NUM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_PAD: tl.constexpr,
    KV_GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KV_GROUPS_PAD: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,
    SM_SCALE: tl.constexpr,
):
    sid = tl.program_id(0)
    kvhid = tl.program_id(1)

    offs_group = tl.arange(0, KV_GROUPS_PAD)
    mask_group = offs_group < KV_GROUPS
    offs_dim = tl.arange(0, HEAD_DIM_PAD)
    mask_dim = offs_dim < HEAD_DIM
    mask_group_dim = mask_group[:, None] & mask_dim[None, :]

    offs_q = offs_group[:, None] * q_stride_kv_g + offs_dim[None, :] * q_stride_kv_d
    # (num_kv_groups_pad, head_dim_pad)
    q = tl.load(
        queries + sid * q_stride_slot + kvhid * q_stride_kv_h + offs_q,
        mask=mask_group_dim,
        other=0,
    )

    active = tl.load(active_mask + sid)
    seq_len = tl.load(position_index + sid) + 1
    num_blocks = tl.cdiv(seq_len, BLOCK_SIZE)

    # Online softmax state for every query head in this KV group. Only a
    # [KV_GROUPS_PAD, BLOCK_SIZE_PAD] score tile is materialized at a time.
    scores_max = tl.full((KV_GROUPS_PAD,), float("-inf"), tl.float32)
    weights_sum = tl.zeros((KV_GROUPS_PAD,), tl.float32)
    acc = tl.zeros((KV_GROUPS_PAD, HEAD_DIM_PAD), tl.float32)

    block_offsets = tl.arange(0, BLOCK_SIZE_PAD)
    block_mask = block_offsets < BLOCK_SIZE
    for logical_block_id in tl.range(0, num_blocks):
        token_offs = logical_block_id * BLOCK_SIZE + block_offsets
        token_mask = active & block_mask & (token_offs < seq_len)

        physical_block_id = tl.load(
            block_table + sid * block_table_stride_slot + logical_block_id
        )

        cache_mask = token_mask[:, None] & mask_dim[None, :]
        # k_layer/v_layer: (block_id, num_kv_heads, block_size, head_dim)
        cache_offsets = (
            physical_block_id * cache_stride_block
            + kvhid * cache_stride_kv_head
            + block_offsets[:, None] * cache_stride_token
            + offs_dim[None, :]
        )

        # keys/values: (block_size_pad, head_dim_pad)
        k_value = tl.load(k_layer + cache_offsets, mask=cache_mask, other=0.0)
        v_value = tl.load(v_layer + cache_offsets, mask=cache_mask, other=0.0)

        # scores: (num_kv_groups_pad, block_size_pad)
        scores = tl.dot(q, tl.trans(k_value)) * SM_SCALE
        # Inactive slots have no valid tokens. Finite dummy scores avoid
        # evaluating -inf - -inf; their masked V loads keep acc equal to zero.
        masked_score = tl.where(active, float("-inf"), 0.0)
        scores = tl.where(token_mask[None, :], scores, masked_score)

        tile_max = tl.max(scores, axis=1)
        new_scores_max = tl.maximum(scores_max, tile_max)
        correction = tl.exp(scores_max - new_scores_max)
        weights = tl.exp(scores - new_scores_max[:, None])

        # Rescale the previous partial sums to the new maximum before adding
        # this KV tile: acc/l now represent every token seen so far.
        acc = acc * correction[:, None]
        acc = tl.dot(weights.to(v_value.dtype), v_value, acc)
        weights_sum = weights_sum * correction + tl.sum(weights, axis=1)
        scores_max = new_scores_max

    result = acc / weights_sum[:, None]
    result = tl.where(active, result, 0.0)

    # out (slot, kv_head, head_group, head_dim)
    offs_out = offs_group[:, None] * HEAD_DIM + offs_dim[None, :]
    tl.store(
        out + (((sid * KV_HEAD_NUM) + kvhid) * KV_GROUPS) * HEAD_DIM + offs_out,
        result,
        mask=mask_group_dim,
    )
