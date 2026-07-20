import triton
import triton.language as tl


@triton.jit
def append_decode_kernel(
    k_layer,
    v_layer,
    keys,
    values,
    position_index,
    active_mask,
    block_table,
    cache_stride_block,
    cache_stride_kv_head,
    cache_stride_token,
    stride_kv_head,
    stride_slot,
    block_table_stride_slot,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,
):
    sid = tl.program_id(0)
    hid = tl.program_id(1)

    active = tl.load(active_mask + sid)
    pos = tl.load(position_index + sid)
    logical_block_id = tl.where(active, pos // BLOCK_SIZE, 0)
    block_offset = tl.where(active, pos % BLOCK_SIZE, 0)
    block_id = tl.load(block_table + sid * block_table_stride_slot + logical_block_id)

    offsets = tl.arange(0, HEAD_DIM_PAD)
    mask = active & (offsets < HEAD_DIM)

    # k_layer/v_layer: (block_id, num_kv_heads, block_size, head_dim)
    cache_offsets = (
        block_id * cache_stride_block
        + hid * cache_stride_kv_head
        + block_offset * cache_stride_token
        + offsets
    )

    # keys/values: (num_kv_heads, slot, head_dim)
    source_offsets = hid * stride_kv_head + sid * stride_slot + offsets
    k_value = tl.load(keys + source_offsets, mask=mask, other=0.0)
    v_value = tl.load(values + source_offsets, mask=mask, other=0.0)

    tl.store(k_layer + cache_offsets, k_value, mask=mask)
    tl.store(v_layer + cache_offsets, v_value, mask=mask)
