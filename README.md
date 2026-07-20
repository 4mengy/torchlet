# Torchlet

**English** | [简体中文](README.zh-CN.md)

Torchlet is a small LLM inference reference project. It is not intended to be a production inference framework. Instead, it keeps the core model and kernel paths visible, then uses named version directories to show how the code evolves through optimization steps.

The current implementation uses a Qwen2.5-style decoder-only Transformer as the running example. It includes RoPE, RMSNorm, GQA, SwiGLU FFN, TransformerBlock, weight loading, and a minimal generation loop.

[**Opne online website**](https://yimdev.github.io/torchlet/)

## Goals

- Show the main LLM inference data flow with as little code as possible.
- Compare kernel optimization ideas across small named version directories.
- Preserve each implementation stage so the structural changes are easy to inspect.
- Prioritize readability and explanatory value before adding more aggressive performance work.

## Implemented Versions

- `v00_full_recompute`: single-request generation with full-sequence recompute.
- `v01_0_ragged_batch`: ragged request batching with `req_indptr`.
- `v01_1_split_gqa`: same behavior as `v01_0`, with the GQA forward path split into readable phases.
- `v02_kv_cache`: simple GQA KV cache with explicit prefill/decode phases.
- `v03_request_states`: explicit request states for waiting, running, and finished requests.
- `v04_continuous_batching`: admit and retire requests while decode continues.
- `v05_decode_slots`: stable slot identities for fixed-width decode batches.
- `v06_static_buffers`: stable decode buffers prepared for graph capture.
- `v07_cuda_graph`: capture and replay of the static decode path.
- `v08_paged_gqa_py`: paged KV cache and readable PyTorch paged GQA.
- `v09_triton_basics`: small Triton kernels for RoPE, RMSNorm, and SwiGLU FFN.
- `v10_triton_paged_gqa`: Triton paged GQA with CUDA Graph decode replay.

See [ROADMAP.md](ROADMAP.md) for the full version path.

## Documentation Site

The repository includes a bilingual static documentation site that explains what each
version introduces, why the change exists, and which files are useful to
compare. It also includes a browser page for side-by-side code comparison across
implemented Versions. The comparison workspace includes changed-file navigation,
split and unified diffs, collapsed context, and shareable URL state.

Build it locally with:

```bash
python3 tools/build_docs.py
```

Then open `docs/_site/index.html`. The Chinese site is available at
`docs/_site/zh/index.html`.

GitHub Pages deployment is configured in `.github/workflows/docs.yml`.

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
from torchlet.v10_triton_paged_gqa.llm import LLM

llm = LLM("Qwen/Qwen2.5-0.5B-Instruct")
outputs = llm.generate([
    "hello, do a simple introduction",
    "what's the nearest star",
])

print(outputs)
```

You can also run the module example:

```bash
python -m torchlet.v10_triton_paged_gqa.llm
```

## Status

Torchlet is still an early reference implementation. The implemented path currently reaches Triton paged GQA with CUDA Graph decode replay.
