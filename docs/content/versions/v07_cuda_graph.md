# v07_cuda_graph

## What This Version Introduces

This version captures the static decode path with CUDA Graph and replays it on later decode steps.

The engine still runs dynamic prefill normally.

## Why Introduce It

Decode often launches many small GPU operations per generated token. Python and CUDA launch overhead can become a large fraction of latency. CUDA Graph reduces that overhead by capturing a fixed sequence of GPU work and replaying it.

## Core Principle

CUDA Graph capture records operations on static decode tensors. Later steps update those tensors in place and replay the captured graph.

The contract is strict:

- Same tensor addresses.
- Same shapes.
- Same control flow in the captured region.
- No dynamic allocation inside the captured path.

## Files To Compare

- `engine.py` for warmup, capture, and replay.
- `scheduler.py` for static decode input mutation.
- `kvcache.py` for fixed slot cache access.
- `layer/gqa.py` for decode attention over static slot-shaped inputs.

## Teaching Tradeoff

This version shows CUDA Graph before paged KV cache exists. That keeps the graph lesson focused, but the cache layout is still not the final serving-shaped memory layout.
