---
title: "Multivariate Correlation Attention"
type: concept
tags:
  - time-series
  - transformer
  - attention
  - multivariate
  - interpretability
created: 2026-05-30
last_updated: 2026-05-30
source_count: 1
confidence: high
status: active
---

# Multivariate Correlation Attention

Multivariate Correlation Attention 是 iTransformer 中将 self-attention 应用于 variate token 维度以显式捕获多变量相关性的机制。与传统做法（attention 建模时间依赖）不同，此机制将 attention score map 视为变量间相关性的可解释表示[^src-itransformer]。

## 机制

给定 $N$ 个 variate token $\mathbf{H} = \{h_0, \ldots, h_N\} \in \mathbb{R}^{N \times D}$，通过线性投影得到 Q, K, V ∈ $\mathbb{R}^{N \times d_k}$。每个 pre-Softmax 分数：

$$A_{i,j} = \left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)_{i,j} \propto q_i^\top k_j$$

由于每个 token 已在特征维度上被 LayerNorm 归一化，$A_{i,j}$ 可揭示变量对 $(i, j)$ 之间的相关性。整个 score map $\mathbf{A} \in \mathbb{R}^{N \times N}$ 展现**多变量相关性结构**，高相关变量在后续 representation interaction 中获得更大权重[^src-itransformer]。

## 可解释性验证

Solar-Energy 数据集上的可视化分析：

1. **浅层注意力图**：与原始输入序列的 Pearson 相关性高度相似
2. **深层注意力图**：逐步接近未来序列的相关性结构
3. **解释**：编码过去（浅层）→ 解码未来（深层）的过程在 FFN 前向传播中完成，注意力图从"看到什么"过渡到"预测什么"

Market 数据集上的额外验证：
- 同一应用类别的变量间注意力分数更高
- 不同应用类别间注意力分数更低
- 注意力图呈现清晰的分块结构，反映变量分组[^src-itransformer]

## 与传统 Temporal Attention 的对比

| 特性 | Temporal Attention | Multivariate Correlation Attention |
|------|-------------------|----------------------------------|
| 作用维度 | 时间步 (T 个 token) | 变量 (N 个 token) |
| 建模目标 | 时间依赖 | 多变量相关性 |
| Score map 含义 | 时间步间关联 | 变量间关联 |
| 可解释性 | 低（时间步数值少语义） | 高（变量相关性有物理意义） |
| 复杂度 | $O(T^2)$ | $O(N^2)$ |
| 排列不变性 | 不当（序列有序） | 恰当（变量无序） |

## 与排列不变性

论文指出，时间序列的变化受序列顺序影响显著，而 attention 机制本质上是排列不变的，在时间维度上使用是不恰当的（Zeng et al., 2023）。在变量维度上使用则更自然——变量间本无严格顺序，排列不变性是合理假设[^src-itransformer]。

## 效率考虑

- 变量数 N 大时 $O(N^2)$ 复杂度 → 可接入高效注意力（Reformer, Flowformer, FlashAttention）
- iTransformer 的高效训练策略：每 batch 随机采样部分变量，降低 N

## 与其他跨维度方法的对比

- **[[crossformer|Crossformer]]**：显式跨维度依赖，通过两阶段注意力（cross-time + cross-dimension），但跨变量 patch 交互引入时间不对齐噪声
- **[[channel-independence]]**：完全独立变量，丢失多变量相关性
- **iTransformer**：保持变量独立嵌入 + attention 显式捕获相关性，简洁且避免噪声

## Connections

- [[inverted-transformer-architecture]] — 此机制是反转架构的核心组件
- [[cross-dimension-dependency]] — 多变量相关性注意力的理论基础
- [[crossformer]] — 另一种显式跨维度方法
- [[itransformer]] — 使用此机制的模型

[^src-itransformer]: [[source-itransformer]]
