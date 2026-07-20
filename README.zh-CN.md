# Torchlet

[English](README.md) | **简体中文**

Torchlet 是一个小型 LLM 推理参考项目。它不以生产级推理框架为目标，而是让核心模型和 kernel 路径保持清晰可见，并通过具名版本目录展示代码如何逐步演进和优化。

当前实现以 Qwen2.5 风格的纯解码器 Transformer 为贯穿示例，包含 RoPE、RMSNorm、GQA、SwiGLU FFN、TransformerBlock、权重加载和最小生成循环。

[**打开在线网站**](https://yimdev.github.io/torchlet/)

## 目标

- 用尽可能少的代码展示 LLM 推理主流程。
- 在小型具名版本目录之间对比 kernel 优化思路。
- 保留每个实现阶段，便于检查结构变化。
- 在引入更激进的性能优化前，优先保证可读性和解释价值。

## 已实现版本

- `v00_full_recompute`：每步完整重算序列的单请求生成。
- `v01_0_ragged_batch`：使用 `req_indptr` 对不规则请求进行批处理。
- `v01_1_split_gqa`：行为与 `v01_0` 相同，但把 GQA 前向路径拆分为清晰阶段。
- `v02_kv_cache`：带显式预填充/解码阶段的简单 GQA KV 缓存。
- `v03_request_states`：为等待、运行和完成请求设置显式状态。
- `v04_continuous_batching`：在解码持续进行时接纳和移除请求。
- `v05_decode_slots`：为固定宽度解码批次提供稳定槽位标识。
- `v06_static_buffers`：为图捕获准备地址稳定的解码缓冲区。
- `v07_cuda_graph`：捕获并重放静态解码路径。
- `v08_paged_gqa_py`：分页 KV 缓存和可读的 PyTorch 分页 GQA。
- `v09_triton_basics`：用于 RoPE、RMSNorm 和 SwiGLU FFN 的小型 Triton kernel。
- `v10_triton_paged_gqa`：Triton 分页 GQA 和 CUDA Graph 解码重放。

完整版本路线请参阅 [ROADMAP.md](ROADMAP.md)。

## 文档站点

仓库包含中英双语静态文档站点，用于说明每个版本引入了什么、为什么需要这项变化，以及哪些文件值得对比。站点还提供浏览器端并排代码对比，支持变更文件导航、并排和统一差异、折叠上下文以及可分享的 URL 状态。

本地构建命令：

```bash
python3 tools/build_docs.py
```

然后打开 `docs/_site/index.html`。中文站点位于 `docs/_site/zh/index.html`。

GitHub Pages 部署配置位于 `.github/workflows/docs.yml`。

## 安装

建议使用 Python 3.12 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

安装开发工具：

```bash
python -m pip install -e ".[dev]"
```

## 示例

```python
from torchlet.v10_triton_paged_gqa.llm import LLM

llm = LLM("Qwen/Qwen2.5-0.5B-Instruct")
outputs = llm.generate([
    "hello, do a simple introduction",
    "what's the nearest star",
])

print(outputs)
```

也可以直接运行模块示例：

```bash
python -m torchlet.v10_triton_paged_gqa.llm
```

## 状态

Torchlet 仍处于早期参考实现阶段。当前已实现路径到达 Triton 分页 GQA 和 CUDA Graph 解码重放。
