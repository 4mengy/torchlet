# v03_request_states

## What This Version Introduces

This version gives requests explicit lifecycle state: waiting, running, and finished.

It also introduces scheduling structure so request movement is no longer hidden inside a single generation loop.

## Why Introduce It

Once requests can be batched and cached, the engine needs to know which requests have not started, which are actively decoding, and which should be returned to the caller. That lifecycle becomes the foundation for continuous batching.

## Core Principle

The model forward path should not decide request lifecycle. The scheduler owns request state, builds model inputs for the current step, and processes generated tokens afterward.

This separates two concerns:

- The model computes logits.
- The scheduler decides who runs next and who is done.

## Files To Compare

- `request.py` for explicit request state.
- `scheduler.py` for request movement.
- `llm.py` for the public generation interface.
- `layer/gqa.py` to confirm attention behavior is still the same cache idea.

## Teaching Tradeoff

The engine is still small and synchronous in spirit. The value here is not production scheduling policy; it is making request state visible before requests start entering and leaving the batch dynamically.
