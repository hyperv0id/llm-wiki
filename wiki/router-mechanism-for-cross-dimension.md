---
title: "Router Mechanism for Cross-Dimension Attention"
type: technique
tags:
  - time-series
  - attention
  - cross-dimension
  - router
  - efficiency
  - transformer
created: 2026-05-30
last_updated: 2026-05-30
source_count: 2
confidence: high
status: active
---

# Router Mechanism for Cross-Dimension Attention

Router 机制是 [[crossformer|Crossformer]] 的 [[two-stage-attention|TSA Layer]] Cross-Dimension Stage 中用于降低维度间注意力复杂度的方法，后被 [[cvpe|CVPE]] 借鉴用于 patch embedding 层 [^src-crossformer-2023][^src-cvpe-2025]。

## 问题

直接对 $D$ 个维度做多头自注意力 (MSA) 复杂度为 $O(D^2 L)$，当 $D$ 较大时（如 Traffic 数据集 $D=862$）不可接受 [^src-crossformer-2023]。

## 机制

设置 $c \ll D$ 个可学习路由向量 $\mathbf{R} \in \mathbb{R}^{L \times c \times d_\text{model}}$，分两步建立维度间的全连接 [^src-crossformer-2023]：

**Step 1 — 聚合 (Gather)**：
$$\mathbf{B}_{i,:} = \text{MSA}^{\text{dim}}_1(\mathbf{R}_{i,:}, \mathbf{Z}_{i,:}, \mathbf{Z}_{i,:})$$
路由向量作 query，所有维度向量作 key/value，从 $D$ 个维度压缩为 $c$ 个聚合表示 $\mathbf{B}$。

**Step 2 — 分发 (Distribute)**：
$$\mathbf{Z}^{\text{dim}}_{i,:} = \text{MSA}^{\text{dim}}_2(\mathbf{Z}_{i,:}, \mathbf{B}_{i,:}, \mathbf{B}_{i,:})$$
维度向量作 query，聚合表示作 key/value，将跨维度信息分发回各维度。

这建立了维度间的全连接：维度 $d_1$ 的信息通过路由器传到 $d_2$。复杂度从 $O(D^2 L)$ 降至 $O(2cD \cdot L) = O(DL)$ [^src-crossformer-2023]。

## 超参数 $c$

路由数 $c$ 控制维度间信息带宽 [^src-crossformer-2023]：
- $c=3$：短期预测足够，但长期预测（τ=720）时信息不足
- $c \geq 5$：长期预测 MSE 趋稳
- 实践中设 $c=10$ 平衡精度与效率

## 在 CVPE 中的应用

[[cvpe|CVPE]] 借鉴了相同的聚合-分发模式，但应用在 patch embedding 层而非全模型 [^src-cvpe-2025]：

| 方面 | Crossformer Router | CVPE Router-Attention |
|------|-------------------|----------------------|
| 位置 | TSA Layer 的 Cross-Dimension Stage | 仅 patch embedding 层 |
| Backbone | 全 CD 架构 | 保留 CI backbone |
| 复杂度 | $O(DL)$ per layer | $O(NP)$ per layer |

## Connections

- 属于：[[two-stage-attention]] — TSA 的 Cross-Dimension Stage
- 属于：[[crossformer]] — 首次提出
- 后续：[[router-attention-for-cvpe]] — CVPE 的具体应用
- 对比：[[adaptive-graph-agent-attention]] — 同样用 agent token 降低复杂度
- 相关：[[cross-dimension-dependency]] — Router 机制建模的依赖类型

[^src-crossformer-2023]: [[source-crossformer-2023]]
[^src-cvpe-2025]: [[source-cvpe-2025]]
