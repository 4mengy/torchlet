# v06_static_buffers

## What This Version Introduces

This version prepares static decode buffers. Decode model inputs keep stable shapes and stable storage addresses across steps.

Prefill remains dynamic; only the decode path is shaped for capture.

## Why Introduce It

CUDA Graph replay requires more than fixed tensor shapes. Captured operations also refer to specific memory addresses. If the engine allocates fresh decode input tensors each step, graph replay cannot safely reuse the captured work.

## Core Principle

Allocate decode input tensors once, then update their contents in place:

- Token IDs.
- Position indices.
- Slot IDs.
- Active masks.

The model sees the same tensor objects each decode step, but their values describe the current requests.

## Files To Compare

- `scheduler.py` for static decode model inputs.
- `engine.py` for reusing decode buffers.
- `forward_params.py` for decode metadata.
- `kvcache.py` for cache positions used by static decode.

## Teaching Tradeoff

This version accepts some awkward buffer plumbing so the CUDA Graph version can focus on capture and replay instead of also introducing static memory discipline.
