# v01_0_ragged_batch

## What This Version Introduces

This version batches requests with different prompt lengths by flattening their tokens into one buffer and using `req_indptr` to mark request boundaries.

Instead of padding every request to the same length, the model receives a compact ragged layout.

## Why Introduce It

Real inference rarely receives one perfectly shaped request at a time. Batching improves utilization, but padding short prompts up to the longest prompt wastes work. Ragged batching shows how to batch variable-length requests while keeping each request's causal attention independent.

## Core Principle

`req_indptr` is a prefix-sum index. Request `i` owns the token slice:

```text
flat_input_ids[req_indptr[i] : req_indptr[i + 1]]
```

GQA loops over those slices, applies causal attention inside each request, and writes results back into the same flat token order.

## Files To Compare

- `forward_params.py` for the new request boundary metadata.
- `request.py` for request containers.
- `llm.py` for batched input construction.
- `layer/gqa.py` for per-request attention over a flat buffer.

## Teaching Tradeoff

The Python loop over requests is intentionally visible. It is not the fastest approach, but it makes the ragged layout concrete before later versions add caching and scheduling.
