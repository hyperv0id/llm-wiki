---
title: "Two-Stage Attention (TSA)"
type: technique
tags:
  - time-series
  - attention
  - cross-dimension
  - cross-time
  - transformer
created: 2026-05-30
last_updated: 2026-05-30
source_count: 2
confidence: high
status: active
---

# Two-Stage Attention (TSA)

TSA Layer 是 [[crossformer|Crossformer]] 的核心注意力机制，分两阶段处理 [[dsw-embedding|DSW embedding]] 输出的 2D 向量阵列，分别捕获跨时间依赖和跨维度依赖 [^src-crossformer-2023]。

## 设计动机

对 2D 向量阵列 $\mathbf{Z} \in \mathbb{R}^{L \times D \times d_\text{model}}$（$L$ 个时间段、$D$ 个维度），直接展平做自注意力导致 $O(D^2 L^2)$ 复杂度 [^src-crossformer-2023]。且时间轴和维度轴有不同语义，不能像图像那样等价处理。因此 TSA 分两阶段分别处理。

## Cross-Time Stage

对每个维度 $d$ 独立做多头自注意力 (MSA) [^src-crossformer-2023]：
$$\hat{\mathbf{Z}}^{\text{time}}_{:,d} = \text{LayerNorm}(\mathbf{Z}_{:,d} + \text{MSA}(\mathbf{Z}_{:,d}, \mathbf{Z}_{:,d}, \mathbf{Z}_{:,d}))$$
$$\mathbf{Z}^{\text{time}}_{:,d} = \text{LayerNorm}(\hat{\mathbf{Z}}^{\text{time}}_{:,d} + \text{MLP}(\hat{\mathbf{Z}}^{\text{time}}_{:,d}))$$

所有 $D$ 个维度共享同一 MSA 层。复杂度 $O(DL^2)$ [^src-crossformer-2023]。

## Cross-Dimension Stage

直接对 $D$ 个维度做 MSA 复杂度为 $O(D^2 L)$，不可接受。TSA 引入 [[router-mechanism-for-cross-dimension|Router 机制]] [^src-crossformer-2023]：

**聚合**：c 个可学习路由向量 $\mathbf{R}$ 作 query，从所有维度聚合信息
$$\mathbf{B}_{i,:} = \text{MSA}^{\text{dim}}_1(\mathbf{R}_{i,:}, \mathbf{Z}^{\text{time}}_{i,:}, \mathbf{Z}^{\text{time}}_{i,:})$$

**分发**：维度向量作 query，路由聚合信息作 key/value
$$\mathbf{Z}^{\text{dim}}_{i,:} = \text{MSA}^{\text{dim}}_2(\mathbf{Z}^{\text{time}}_{i,:}, \mathbf{B}_{i,:}, \mathbf{B}_{i,:})$$

所有时间步 $L$ 共享 $\text{MSA}^{\text{dim}}_1, \text{MSA}^{\text{dim}}_2$。复杂度从 $O(D^2 L)$ 降至 $O(DL)$ [^src-crossformer-2023]。

## 总体

$$\mathbf{Y} = \mathbf{Z}^{\text{dim}} = \text{TSA}(\mathbf{Z})$$

总复杂度 $O(DL^2 + DL) = O(DL^2)$。每个输出向量与所有其他段相连，跨时间和跨维度依赖均被捕获 [^src-crossformer-2023]。

## 与 CVPE Router-Attention 的关系

[[cvpe|CVPE]] 的 Router-Attention 直接借鉴了 TSA 的 Cross-Dimension Stage 的 Router 机制，但仅应用于 patch embedding 层而非全模型各层 [^src-cvpe-2025]。

## Connections

- 属于：[[crossformer]] — 核心注意力组件
- 包含：[[router-mechanism-for-cross-dimension]] — Cross-Dimension Stage 的核心
- 相关：[[cross-dimension-dependency]] — TSA 显式建模的依赖
- 相关：[[dsw-embedding]] — TSA 处理 DSW 输出
- 后续：[[router-attention-for-cvpe]] — CVPE 借鉴了 Router 机制

[^src-crossformer-2023]: [[source-crossformer-2023]]
[^src-cvpe-2025]: [[source-cvpe-2025]]
