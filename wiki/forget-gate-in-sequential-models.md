---
title: "Forget Gate in Sequential Models"
type: technique
tags:
  - mamba
  - gating-mechanism
  - positional-encoding
  - sequential-modeling
created: 2026-05-08
last_updated: 2026-05-08
source_count: 1
confidence: medium
status: active
---

# Forget Gate in Sequential Models

**遗忘门（Forget Gate）** 是 Mamba 选择性状态空间模型中的核心机制，通过输入依赖的指数衰减控制历史信息的保留程度。Han et al. (NeurIPS 2024) 证明遗忘门可被位置编码替代，从而在保持功能的同时实现并行计算[^src-demystify-mamba-linear-attention-2024]。

## Mamba 中的遗忘门

在 Mamba 的 SSM 递推中，遗忘门 $\widetilde{A}_i$ 逐元素乘以上一时刻的隐藏状态[^src-demystify-mamba-linear-attention-2024]：

$$h_i = \widetilde{A}_i \odot h_{i-1} + B_i(\Delta_i \odot x_i)$$

其中 $\widetilde{A}_i = \text{diag}(\exp(\Delta_i A))$，其值域为 $(0,1)$，实现输入依赖的指数衰减。

## 累积衰减效应

展开递推后，历史 token $j$ 对当前输出 $i$ 的贡献被衰减因子加权[^src-demystify-mamba-linear-attention-2024]：

$$y_i = C_i \sum_{j=1}^i (\prod_{k=j+1}^i \widetilde{A}_k) B_j (\Delta_j \odot x_j) + D \odot x_i$$

其中 $\prod_{k=j+1}^i \widetilde{A}_k$ 对距离 $i-j$ 越大的 token 施加越强的衰减，形成位置依赖的注意力权重。

## 遗忘门的双重功能

消融实验表明遗忘门提供两种关键功能[^src-demystify-mamba-linear-attention-2024]：

1. **局部偏置（Local Bias）**：自然地对近邻 token 赋予更高权重，类似于 ALiBi 的距离衰减
2. **输入依赖的位置信息**：衰减强度随输入内容变化，实现内容感知的位置建模

## 位置编码替代方案

由于遗忘门需要循环计算（吞吐量从 1152 降至 743 im/s），Han et al. 提出使用位置编码替代[^src-demystify-mamba-linear-attention-2024]：

- **LePE（Local Enhanced Positional Encoding）**：提供局部偏置
- **CPE（Conditional Positional Encoding）**：提供条件化的位置信息
- **RoPE（Rotary Position Embedding）**：提供相对位置编码

这些替代方案在 [[mila|MILA]] 中实现，在保持并行计算的同时达到或超越原始 Mamba 的性能。

## 与 ALiBi 的关系

遗忘门的局部衰减与 [[linear-attention-bias|ALiBi]] 的线性偏置功能相似，但存在关键差异[^src-demystify-mamba-linear-attention-2024]：

- ALiBi 是加性偏置，遗忘门是乘性衰减
- ALiBi 是固定的，遗忘门是输入依赖的
- 遗忘门通过指数形式实现更灵活的衰减模式

## 相关页面

- [[mamba|Mamba]] — 选择性状态空间模型
- [[mila|MILA]] — 使用位置编码替代遗忘门的线性注意力模型
- [[linear-attention-unified-framework|Mamba ↔ Linear Attention 统一框架]]
- [[linear-attention-bias|线性注意力偏置（ALiBi）]]
- [[generalized-positional-encoding-framework|广义位置编码框架（GPE）]]

[^src-demystify-mamba-linear-attention-2024]: [[source-demystify-mamba-linear-attention-2024]]
