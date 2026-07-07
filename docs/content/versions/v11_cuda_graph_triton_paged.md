# v11_cuda_graph_triton_paged

## What This Version Introduces

This planned version captures the Triton paged decode path with CUDA Graph.

It combines the two major late-roadmap ideas: paged KV cache and graph replay.

## Why Introduce It

Paged attention improves the memory layout, while CUDA Graph reduces launch overhead. In a decode loop, both matter. This version shows what must become static once the hot path uses custom kernels.

## Core Principle

The graph captures a fixed Triton decode launch shape and fixed tensor addresses. Runtime changes should happen by mutating existing tensors:

- Input token buffer.
- Position buffer.
- Active mask.
- Block table contents.
- Cache contents.

The captured operations stay the same; the data they read changes in place.

## Files To Compare

- The v11 engine against `v07_cuda_graph/engine.py`.
- The v11 Triton paged attention path against the v10 kernel.
- Static buffer and block-table handling against `v08_paged_gqa_py/scheduler.py`.

## Teaching Tradeoff

This is the most constraint-heavy core version. The documentation should call out which constraints come from CUDA Graph, which come from Triton launch shape, and which come from paged cache layout.
