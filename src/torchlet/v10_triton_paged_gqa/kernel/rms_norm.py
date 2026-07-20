import triton
import triton.language as tl


# H_PAD = triton.next_power_of_2(H)
@triton.jit
def rms_norm_kernel(x, w, eps, out, H, H_PAD: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, H_PAD)
    mask = offs < H
    x_ptr = x + pid * H + offs

    x_m = tl.load(x_ptr, mask=mask, other=0).to(tl.float32)

    w_ptr = w + offs
    w_m = tl.load(w_ptr, mask=mask, other=0).to(tl.float32)

    avg = tl.sum(x_m * x_m, axis=0) / H + eps
    z = x_m * tl.rsqrt(avg) * w_m

    out_ptr = out + pid * H + offs
    tl.store(out_ptr, z, mask=mask)
