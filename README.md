# Torchlet

Torchlet is a small educational LLM inference project. It is not intended to be a production inference framework. Instead, it keeps the core model and kernel paths visible, then uses separate `vX` implementations to show how the code evolves through optimization steps.

The current implementation uses a Qwen2.5-style decoder-only Transformer as the running example. It includes RoPE, RMSNorm, GQA, SwiGLU FFN, TransformerBlock, weight loading, and a minimal generation loop.

## Goals

- Show the main LLM inference data flow with as little code as possible.
- Compare kernel optimization ideas across `v0`, `v1`, and future `v2`/`v3`/... versions.
- Preserve each implementation stage so the structural changes are easy to inspect.
- Prioritize readability and learning value before adding more aggressive performance work.

## Version Layout

```text
src/torchlet/
  v0/
    layer/
    model/
    llm.py
  v1/
    layer/
    model/
    request.py
    forward_params.py
    llm.py
```

Each `vX` directory represents a relatively independent implementation stage. Imports inside a version package use relative imports where practical, which makes it easy to copy a version and continue evolving it.

### v0

`v0` is the most direct implementation:

- Single-request generation.
- Inputs keep the `[batch, seq_len]` shape.
- Each generation step runs a full forward pass over the whole sequence.
- GQA, RoPE, RMSNorm, and FFN are written close to the underlying formulas.

This version is useful for understanding the model structure and the basic inference flow.

### v1

`v1` starts showing the data-layout changes needed for batched inference kernels:

- Multiple requests can generate at the same time.
- Active request tokens are flattened into a single one-dimensional token buffer.
- `req_indptr` describes the request boundaries inside the flat token buffer.
- The attention kernel splits work by request so different requests cannot attend to each other.
- With `last_token_only=True`, logits are gathered from the last token of each request.

This version is useful for understanding ragged batches, request batching, and runtime kernel parameters.

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
from torchlet.v1.llm import LLM

llm = LLM("Qwen/Qwen2.5-0.5B-Instruct")
outputs = llm.generate([
    "hello, do a simple introduction",
    "what's the nearest star",
])

print(outputs)
```

You can also run the module example:

```bash
python -m torchlet.v1.llm
```

## Status

Torchlet is still an early educational implementation. Future versions can explore KV cache, paged attention, operator fusion, quantization, and custom CUDA/Triton kernels.
