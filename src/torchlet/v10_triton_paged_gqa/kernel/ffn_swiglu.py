import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_D": 32, "BLOCK_H": 64},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_D": 32, "BLOCK_H": 64},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_D": 32, "BLOCK_H": 64},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_D": 32, "BLOCK_H": 128},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_D": 32, "BLOCK_H": 128},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_D": 64, "BLOCK_H": 64},
            num_warps=4,
            num_stages=4,
        ),
    ],
    key=["M", "D", "H"],
)
@triton.jit
def down_kernel(
    z_ptr,
    wd_ptr,
    out_ptr,
    M,
    D,
    H,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # M: tokens, D: FFN intermediate size, H: hidden size
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_m = offs_m < M
    mask_h = offs_h < H

    out_acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        z_ptrs = z_ptr + offs_m[:, None] * D + offs_d[None, :]
        z_mask = mask_m[:, None] & mask_d[None, :]

        # nn.Linear stores wd as [H, D]; load its logical transpose [D, H].
        wd_ptrs = wd_ptr + offs_h[None, :] * D + offs_d[:, None]
        wd_mask = mask_d[:, None] & mask_h[None, :]

        z = tl.load(z_ptrs, mask=z_mask, other=0.0)
        wd = tl.load(wd_ptrs, mask=wd_mask, other=0.0)
        out_acc += tl.dot(z, wd)

    out_ptrs = out_ptr + offs_m[:, None] * H + offs_h[None, :]
    out_mask = mask_m[:, None] & mask_h[None, :]
    tl.store(out_ptrs, out_acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_H": 32, "BLOCK_D": 64},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_H": 32, "BLOCK_D": 64},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_H": 32, "BLOCK_D": 64},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_H": 32, "BLOCK_D": 128},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_H": 32, "BLOCK_D": 128},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_H": 64, "BLOCK_D": 64},
            num_warps=4,
            num_stages=4,
        ),
    ],
    key=["M", "H", "D"],
)
@triton.jit
def swiglu_kernel(
    x_ptr,
    wg_ptr,
    wu_ptr,
    z_ptr,
    M,
    H,
    D,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # M: tokens, H: hidden size, D: FFN intermediate size.
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = offs_m < M
    mask_d = offs_d < D

    gate_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for h_start in range(0, H, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        x_ptrs = x_ptr + offs_m[:, None] * H + offs_h[None, :]
        x_mask = mask_m[:, None] & mask_h[None, :]
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # nn.Linear stores wg/wu as [D, H]; load their logical transpose [H, D].
        wg_ptrs = wg_ptr + offs_d[None, :] * H + offs_h[:, None]
        w_mask = mask_h[:, None] & mask_d[None, :]
        wg = tl.load(wg_ptrs, mask=w_mask, other=0.0)

        wu_ptrs = wu_ptr + offs_d[None, :] * H + offs_h[:, None]
        wu = tl.load(wu_ptrs, mask=w_mask, other=0.0)

        gate_acc += tl.dot(x, wg)
        up_acc += tl.dot(x, wu)

    z = gate_acc * tl.sigmoid(gate_acc) * up_acc

    z_ptrs = z_ptr + offs_m[:, None] * D + offs_d[None, :]
    z_mask = mask_m[:, None] & mask_d[None, :]
    tl.store(z_ptrs, z, mask=z_mask)
