# v01_1_split_gqa

## What This Version Introduces

This version keeps the same behavior as `v01_0_ragged_batch`, but splits the GQA forward path into named phases.

The attention code becomes easier to inspect before KV cache adds a second execution mode.

## Why Introduce It

Optimization work is much harder when a core function is one long block of tensor reshapes, RoPE, masking, and projection logic. This checkpoint creates a cleaner baseline before the next version changes behavior.

## Core Principle

Grouped Query Attention has separable conceptual steps:

- Project Q, K, and V.
- Reshape heads.
- Apply RoPE to query and key.
- Group query heads by KV head.
- Run causal attention per request.
- Project the context vector back to model width.

Giving those steps names lowers the cost of comparing future diffs.

## Files To Compare

- `layer/gqa.py` against `v01_0_ragged_batch/layer/gqa.py`.

## Teaching Tradeoff

This version is a readability checkpoint, not a performance checkpoint. It teaches that some version steps are valuable because they make the next change understandable.
