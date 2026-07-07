# v05_decode_slots

## What This Version Introduces

This version adds fixed decode slots. Running decode requests are assigned stable slot IDs, and decode inputs can be shaped around the maximum number of slots.

## Why Introduce It

Continuous batching makes the active request set dynamic. CUDA Graph and static decode buffers need the opposite: stable shapes and predictable tensor addresses.

Decode slots are the bridge between those worlds.

## Core Principle

A slot is a stable position in the decode batch. A request can enter a slot, generate tokens over multiple steps, then release the slot when it finishes.

The engine can build decode tensors sized by `max_decode_slots`, while an active mask or dummy tokens represent unused slots.

## Files To Compare

- `scheduler.py` for slot assignment and release.
- `engine.py` for fixed-width decode input construction.
- `forward_params.py` for slot metadata.
- `layer/gqa.py` for decode attention indexed by slot/request mapping.

## Teaching Tradeoff

Slots add bookkeeping and may carry inactive lanes. That is the price of making the decode step more static.
