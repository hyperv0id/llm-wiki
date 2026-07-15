---
title: "Mixed Graph Spatiotemporal Modeling"
type: concept
tags:
  - spatial-temporal
  - graph-signal-processing
  - traffic-forecasting
created: 2026-07-16
last_updated: 2026-07-16
source_count: 1
confidence: medium
status: active
---

# Mixed Graph Spatiotemporal Modeling

**Mixed Graph Spatiotemporal Modeling** 是一种用混合图（同时包含无向边和有向边）来统一表示时空数据中空间和时间依赖关系的建模框架[^src-lightweight-mixed-graph-unrolling]。

## 核心思想

时空数据（如交通、气象）包含两类根本不同的节点间关系[^src-lightweight-mixed-graph-unrolling]：

1. **空间相关性** — 地理上邻近的观测站表现出相似的信号模式，这种关系是**对称/无向的**。如两个相邻路口的车流量高度相关。

2. **时间顺序关系** — 过去的观测影响未来的值，这种关系是**非对称/有向的**。$t$ 时刻的交通状态决定了 $t+1$ 时刻的状态，反之不成立。

混合图通过在一个统一框架中同时包含两类边来建模这两种关系[^src-lightweight-mixed-graph-unrolling]：

- **无向图 $G^u$**：空间维度，边 $(i,j) \in \mathcal{E}^u$ 连接同一时刻的空间邻居
- **有向无环图 (DAG) $G^d$**：时间维度，边 $[i,j] \in \mathcal{E}^d$ 从时刻 $\tau$ 的节点指向未来窗口 $\tau+1,\dots,\tau+W$ 的同一节点

两者组合为 $N \times (T+S+1)$ 节点的 product graph。

## 与纯无向图的区别

仅使用无向图（如 Thuc et al., 2024 对静态图像插值）无法区分空间相似性和时间因果性[^src-lightweight-mixed-graph-unrolling]。有向边天然编码了时间箭头 — 过去影响未来，而非反之。

## 数学处理

- 无向图 $G^u$ 上使用标准 [[graph-laplacian-regularizer|GLR]]：$\mathbf{x}^\top \mathbf{L}^u \mathbf{x}$
- 有向图 $G^d$ 上使用 [[directed-graph-laplacian-regularizer|DGLR]] 和 [[directed-graph-total-variation|DGTV]]

## 参考文献

[^src-lightweight-mixed-graph-unrolling]: [[source-lightweight-mixed-graph-unrolling]]
