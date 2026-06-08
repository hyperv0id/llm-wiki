---
title: "Dual-Dimension Feature Disentanglement"
type: technique
tags:
  - feature-disentanglement
  - spatio-temporal
  - low-rank
  - encoder
  - retrieval-augmented
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Dual-Dimension Feature Disentanglement

双维度特征解耦（Dual-Dimension Feature Disentanglement）是 [[rast|RAST]] 框架的核心设计原则，通过将时空数据显式分解为独立的时间和空间嵌入流，为后续的双维度检索增强奠定基础。[^src-rast]

## 动机

时空预测涉及两类异质动态：[^src-rast]
- **时间依赖性**：多尺度周期性（如高峰时段模式）
- **空间相关性**：局部拓扑驱动的相互作用

在传统架构中，这两类信息被联合编码为高维纠缠嵌入，不仅增加了细粒度模式学习的难度，还提高了存储和检索上下文信息的成本。[^src-rast]

## 数学基础

受 LoRA 低秩分解启发，传统时空嵌入 $H \in \mathbb{R}^{N \times T \times d}$ 可近似为：[^src-rast]

$$H \approx UV^\top, \quad U \in \mathbb{R}^{T \times d}, V \in \mathbb{R}^{N \times d}$$

这一分解将存储和检索复杂度从 $O(N \cdot T \cdot d)$ 降至 $O((N + T) \cdot r)$，其中 $r$ 为解耦后的特征秩。[^src-rast]

## 编码器设计

### 时间编码器

使用带扩张的 1D 卷积捕获多尺度时间模式：[^src-rast]

$$E_{tp} = \sigma(\text{Conv2D}(X)) \in \mathbb{R}^{B \times N \times D_{tp}}$$

核权重采用 Kaiming 正态初始化确保训练稳定性。

### 空间编码器

采用图卷积操作适配道路网络拓扑：[^src-rast]

$$E_{sp} = \sigma(W_{sp}(X, G)) \in \mathbb{R}^{B \times N \times D_{sp}}$$

变换矩阵 $W_{sp}$ 使用 Xavier 均匀分布初始化。

## 对检索的影响

双维度设计使得 RAST 维护两个紧凑的专用检索库（参见 [[spatio-temporal-retrieval-store]]），每个针对单一维度优化：[^src-rast]
- 时间查询匹配时间历史模式
- 空间查询匹配空间历史模式

这避免了存储全时空张量的立方增长，同时通过交叉注意力融合恢复原始时空交互。[^src-rast]

## 消融验证

移除空间编码器导致 MAE 退化 17.2%，移除时间编码器导致退化 21.2%，证明双流编码对捕获时空依赖不可或缺。[^src-rast]

## 与其他解耦方法的对比

| 方法 | 解耦策略 | 动机 | 领域 |
|------|----------|------|------|
| RAST | 时间/空间双流 | 为双维度检索提供基础 | 交通预测 |
| [[factost|FactoST]] | 时间预训练+空间适配 | 域不变 vs 域特定解耦 | 通用 STF |
| [[std-mae|STD-MAE]] | 时空解耦 MAE 预训练 | 避免信息泄漏 | 预训练 |

[^src-rast]: [[source-rast]]