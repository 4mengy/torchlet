# v02_kv_cache

## What This Version Introduces

This version adds a simple KV cache and separates generation into prefill and decode phases.

Prefill processes the prompt and stores keys and values. Decode processes one new token per request and attends over cached history.

## Why Introduce It

Full recompute wastes work because old tokens keep producing the same keys and values. KV cache stores that historical state so decode can focus on the new token.

This is the first serving-shaped optimization in the roadmap.

## Core Principle

For each layer and request, the cache stores:

```text
K: [num_kv_heads, cached_seq_len, head_dim]
V: [num_kv_heads, cached_seq_len, head_dim]
```

During decode, the model projects K and V for the new token, appends them to the cache, and computes attention from the new query to the cached prefix.

## Files To Compare

- `kvcache.py` for the first cache structure.
- `forward_params.py` for cache and phase metadata.
- `layer/gqa.py` for the prefill/decode branch.
- `llm.py` for generation flow changes.

## Teaching Tradeoff

The cache is simple and request-owned. That is easy to read, but it does not yet solve scheduling, fixed decode shape, memory fragmentation, or graph capture.
