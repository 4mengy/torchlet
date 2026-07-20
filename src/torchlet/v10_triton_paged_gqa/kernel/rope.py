import triton
import triton.language as tl


@triton.jit
def rope_kernel(
    x,
    freqs,
    position_index,
    out,
    x_stride_h,
    x_stride_t,
    out_stride_h,
    out_stride_t,
    half_head_dim,
    half_head_dim_pad: tl.constexpr,
):
    hid = tl.program_id(0)
    tid = tl.program_id(1)

    tok_pos = tl.load(position_index + tid)

    offs = tl.arange(0, half_head_dim_pad)
    freqs_mask = offs < half_head_dim
    freqs_ptr = freqs + tok_pos * half_head_dim + offs
    freqs_m = tl.load(freqs_ptr, mask=freqs_mask, other=0)

    cos = tl.cos(freqs_m)
    sin = tl.sin(freqs_m)

    mask = offs < half_head_dim
    x_base_ptr = hid * x_stride_h + tid * x_stride_t
    x1 = tl.load(x + x_base_ptr + offs, mask=mask, other=0)
    x2 = tl.load(x + x_base_ptr + offs + half_head_dim, mask=mask, other=0)

    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin

    out_base_ptr = hid * out_stride_h + tid * out_stride_t
    tl.store(out + out_base_ptr + offs, y1, mask=mask)
    tl.store(out + out_base_ptr + offs + half_head_dim, y2, mask=mask)
