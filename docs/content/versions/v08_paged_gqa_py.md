# v08_paged_gqa_py

## What This Version Introduces

This version replaces slot-owned contiguous KV storage with a paged KV cache implemented in readable PyTorch.

Requests receive logical blocks that map to physical cache blocks through a block table.

## Why Introduce It

Contiguous per-request KV cache is easy to understand, but it becomes awkward when requests have different lengths and finish at different times. Paged KV cache makes memory reusable at block granularity and prepares the data layout used by paged attention kernels.

## Core Principle

The request sees a logical sequence of cache blocks:

```text
logical block 0, logical block 1, logical block 2, ...
```

The cache stores physical blocks in a pool. A block table maps each request slot and logical block index to a physical block ID.

Attention reconstructs the request's logical K/V order by following the table.

## Files To Compare

- `kvcache.py` for block pool allocation and physical storage.
- `scheduler.py` for block table construction.
- `layer/gqa.py` for PyTorch paged attention reads.
- `engine.py` for how paged cache coexists with CUDA Graph decode.

## Teaching Tradeoff

The implementation still uses Python loops and PyTorch gathers so the mapping stays visible. The next major step is to move this same idea into Triton kernels.
