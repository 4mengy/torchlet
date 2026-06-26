# Torchlet

Torchlet is a small educational LLM inference project. It is not intended to be a production inference framework. Instead, it keeps the core model and kernel paths visible, then uses named version directories to show how the code evolves through optimization steps.

The current implementation uses a Qwen2.5-style decoder-only Transformer as the running example. It includes RoPE, RMSNorm, GQA, SwiGLU FFN, TransformerBlock, weight loading, and a minimal generation loop.

## Goals

- Show the main LLM inference data flow with as little code as possible.
- Compare kernel optimization ideas across small named version directories.
- Preserve each implementation stage so the structural changes are easy to inspect.
- Prioritize readability and learning value before adding more aggressive performance work.

## Implemented Versions

- `v00_full_recompute`: single-request generation with full-sequence recompute.
- `v01_0_ragged_batch`: ragged request batching with `req_indptr`.
- `v01_1_split_gqa`: same behavior as `v01_0`, with the GQA forward path split into readable phases.
- `v02_kv_cache`: simple GQA KV cache with explicit prefill/decode phases.

See [ROADMAP.md](ROADMAP.md) for the full planned version path.

## Installation

Python 3.12+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For development tools:

```bash
python -m pip install -e ".[dev]"
```

## Example

```python
from torchlet.v02_kv_cache.llm import LLM

llm = LLM("Qwen/Qwen2.5-0.5B-Instruct")
outputs = llm.generate([
    "hello, do a simple introduction",
    "what's the nearest star",
])

print(outputs)
```

You can also run the module example:

```bash
python -m torchlet.v02_kv_cache.llm
```

## Status

Torchlet is still an early educational implementation. Future versions can explore KV cache, paged attention, operator fusion, quantization, and custom CUDA/Triton kernels.
