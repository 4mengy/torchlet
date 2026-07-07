# v10_triton_paged_gqa

## What This Version Introduces

This planned version moves paged GQA attention from readable PyTorch into Triton.

The block table and paged KV cache idea should remain the same, but the hot attention loop moves into a custom kernel.

## Why Introduce It

The PyTorch version makes the memory mapping clear, but Python loops and generic tensor operations are not the final form for fast decode attention. Triton can express the fixed access pattern more directly.

## Core Principle

The kernel should use the block table to translate logical token positions into physical KV blocks. For each decode query, it loads the relevant K/V tiles, applies the valid-token mask, computes attention scores, and writes the context vector.

The important mental model is:

```text
request slot + logical block -> physical block -> K/V tile
```

## Files To Compare

- The Triton paged GQA kernel against `v08_paged_gqa_py/layer/gqa.py`.
- The cache layout against `v08_paged_gqa_py/kvcache.py`.
- The scheduler block table construction against the v08 version.

## Teaching Tradeoff

The kernel should initially optimize for correctness and shape clarity. Aggressive fusion and autotuning can wait until the paged attention mapping is obvious.
