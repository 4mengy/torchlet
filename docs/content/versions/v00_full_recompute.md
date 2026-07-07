# v00_full_recompute

## What This Version Introduces

This is the smallest useful baseline: a Qwen2.5-style decoder-only model that serves one request and recomputes the whole sequence on every generation step.

It includes the core model pieces that later versions keep reusing: token embedding, RoPE, RMSNorm, grouped query attention, SwiGLU, the Transformer block, weight loading, and greedy generation.

## Why Introduce It

A readable baseline gives every later optimization something concrete to improve. Without this version, KV cache, batching, slots, paged storage, and CUDA Graph would appear as isolated techniques instead of responses to real bottlenecks.

## Core Principle

Autoregressive generation repeatedly appends one token, then runs the model on the prompt plus all generated tokens so far. Causal attention keeps each position from looking into the future, but the implementation still recomputes keys and values for old tokens each step.

That makes the implementation simple and honest, but it also makes the cost grow quickly as the sequence length increases.

## Files To Compare

- `llm.py` for the single-request generation loop.
- `layer/gqa.py` for the first complete GQA implementation.
- `model/qwen2_5.py` for the model skeleton that later versions preserve.

## Teaching Tradeoff

This version favors clarity over efficiency. It is deliberately too slow for serving, which makes the next question natural: how do we avoid recomputing historical attention state?
