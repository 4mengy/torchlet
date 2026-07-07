# v04_continuous_batching

## What This Version Introduces

This version adds a tiny continuous batching engine. New requests can enter while existing requests continue decoding, and finished requests can leave without stopping the whole batch.

## Why Introduce It

Static batches are easy, but serving traffic is not static. If the engine waits for every request in a batch to finish before admitting new work, fast requests get blocked behind slow ones and GPU utilization falls.

## Core Principle

Each step selects a mix of work from the waiting and running pools:

- Prefill work starts new requests.
- Decode work advances running requests by one token.
- Finished requests are drained and completed.

The important change is that the batch is now a stream of request state transitions, not a fixed list.

## Files To Compare

- `engine.py` for the step loop.
- `scheduler.py` for admitting and draining requests.
- `request.py` for lifecycle fields.
- `kvcache.py` for cache ownership while requests move through the engine.

## Teaching Tradeoff

The policy is intentionally simple. It teaches the mechanics of continuous batching before introducing fixed slots and static buffers.
