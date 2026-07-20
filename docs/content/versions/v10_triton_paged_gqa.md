# v10_triton_paged_gqa

## What This Version Introduces

This version moves paged GQA attention from readable PyTorch into Triton and captures the fixed decode path with CUDA Graph.

The block table and paged KV cache layout remain the same, while the hot attention loop moves into a custom kernel whose decode launch can be replayed.

## Why Introduce It

The PyTorch version makes the memory mapping clear, but Python loops and generic tensor operations are not the final form for fast decode attention. Triton can express the fixed access pattern more directly, while CUDA Graph avoids repeatedly paying Python and launch overhead in the decode loop.

## Core Principle

The kernel uses the block table to translate logical token positions into physical KV blocks. For each decode query, it loads the relevant K/V tiles incrementally, applies the valid-token mask, and uses online softmax to maintain the running maximum, exponential sum, and Value-weighted accumulator. This produces the same context vector as full softmax without materializing the complete attention-score matrix.

The important mental model is:

```text
request slot + logical block -> physical block -> K/V tile
                                                 -> update m / l / acc
```

Each KV tile applies the following recurrence, where `m`, `l`, and `acc` are the running maximum score, exponential sum, and unnormalized output:

```text
m_new = max(m, max(scores_tile))
alpha = exp(m - m_new)
p     = exp(scores_tile - m_new)
l     = alpha * l + sum(p)
acc   = alpha * acc + p @ V_tile
```

After all valid KV tokens have been visited, the final output is `acc / l`.

The graph captures a fixed Triton decode launch shape and fixed tensor addresses. Runtime changes should happen by mutating existing tensors:

- Input token buffer.
- Position buffer.
- Active mask.
- Block table contents.
- Cache contents.

The captured operations stay the same; the data they read changes in place.

## Files To Compare

- The Triton paged GQA kernel against `v08_paged_gqa_py/layer/gqa.py`.
- The engine and CUDA Graph path against `v07_cuda_graph/engine.py`.
- The cache layout against `v08_paged_gqa_py/kvcache.py`.
- The scheduler block table construction against the v08 version.

## Remaining Tradeoff

This is the most constraint-heavy core version. The implementation should keep correctness and shape clarity ahead of aggressive fusion and autotuning, and the documentation should distinguish constraints from CUDA Graph, Triton launch shape, and paged cache layout.
