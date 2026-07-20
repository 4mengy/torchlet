# v10_triton_paged_gqa

## 本版本引入什么

这个计划版本把分页 GQA 注意力从易读的 PyTorch 迁移到 Triton，并使用 CUDA Graph 捕获固定解码路径。

块表和分页 KV 缓存的思路保持不变，但热点注意力循环进入自定义 kernel，其解码启动可以被重放。

## 为什么引入

PyTorch 版本清楚呈现了内存映射，但 Python 循环和通用张量操作并不是快速解码注意力的最终形态。Triton 可以更直接地表达固定访问模式，CUDA Graph 则避免在解码循环里反复支付 Python 和 kernel 启动开销。

## 核心原则

Kernel 使用块表把逻辑词元位置转换为物理 KV 块。对于每个解码查询，它逐块加载相应 K/V，应用有效词元掩码，并通过 Online Softmax 维护当前最大值、指数和与 Value 加权和。这样无需物化完整注意力分数矩阵，就能写出与完整 Softmax 等价的上下文向量。

重要的理解模型是：

```text
请求槽位 + 逻辑块 -> 物理块 -> K/V 分块
                                   -> 更新 m / l / acc
```

每个 KV 分块使用下面的递推，其中 `m`、`l` 和 `acc` 分别表示当前最大分数、指数和与未归一化输出：

```text
m_new = max(m, max(scores_tile))
alpha = exp(m - m_new)
p     = exp(scores_tile - m_new)
l     = alpha * l + sum(p)
acc   = alpha * acc + p @ V_tile
```

遍历完有效 KV 词元后，最终输出为 `acc / l`。

CUDA Graph 捕获固定的 Triton 解码启动形状和张量地址。运行时变化通过原地修改已有张量完成：

- 输入词元缓冲区。
- 位置缓冲区。
- 活跃掩码。
- 块表内容。
- 缓存内容。

被捕获的操作保持不变，它们读取的数据会原地更新。

## 建议对比的文件

- 将 Triton 分页 GQA kernel 与 `v08_paged_gqa_py/layer/gqa.py` 对比。
- 将引擎和 CUDA Graph 路径与 `v07_cuda_graph/engine.py` 对比。
- 将缓存布局与 `v08_paged_gqa_py/kvcache.py` 对比。
- 将调度器块表构造与 v08 对比。

## 保留的权衡

这是核心路线中约束最多的版本。实现应优先保证正确性和形状清晰度，而不是激进融合与自动调优；文档也应区分哪些约束来自 CUDA Graph、Triton 启动形状和分页缓存布局。
