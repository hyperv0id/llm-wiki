---
title: "分叉薛定谔桥"
type: concept
tags:
  - schrodinger-bridge
  - branched-transport
  - multimodal
  - cell-differentiation
  - optimal-transport
created: 2026-06-16
last_updated: 2026-06-16
source_count: 1
confidence: medium
status: active
---

# 分叉薛定谔桥

Branched Schrödinger Bridge（Tang et al. 2026）是 [[schrodinger-bridge|薛定谔桥]] 的一种扩展，用于处理终端分布具有多模态（multimodal）结构的场景。与将全部概率质量拉向单一目标的 classical SB 不同，branched SB 允许轨迹根据模态归属自然**分叉**，避免 mode collapse。[^src-schrodinger-bridges-generative-modeling]

## 分支结构

系统由 $K+1$ 个支路（branch）组成：

- **1 个主支路**（$k=0$），承载共享的早期动力学
- **$K$ 个次级支路**（$k=1,\ldots,K$），对应 $K$ 个目标 mode

每个支路有自己独立的 control drift $u_k(x,t)$，次级支路还拥有自己的 growth rate $g_k(x,t)$（允许质量在各支路之间分配）。[^src-schrodinger-bridges-generative-modeling]

## 全局 Fokker-Planck 方程

设 $p_{t,k}$ 为第 $k$ 支路在时刻 $t$ 的质量密度，全局密度 $p_t = \sum_{k=0}^K p_{t,k}$。全局 Fokker-Planck 方程为：

$$\partial_t p_t = -\nabla\cdot(p_t(f+\sigma_t u)) + \frac{\sigma_t^2}{2}\Delta p_t + \sum_{k=1}^K g_k p_{t,k}$$

其中 growth terms $g_k p_{t,k}$ 描述第 $k$ 支路的质量变化（分支产生）。[^src-schrodinger-bridges-generative-modeling]

## 有效控制与支路权重

由于各支路的 control $u_k$ 不同，全局有效 drift 为各支路按质量加权的平均：

$$u(x,t) = \frac{1}{p_t(x)} \sum_{k=0}^K w_{t,k}(x) u_k(x,t) p_{t,k}(x)$$

支路的累积权重由 growth rate 积分得到：

$$w_{t,0} = 1 + \int_0^t g_0 ds, \quad w_{t,k} = \int_0^t g_k ds \quad (k=1,\ldots,K)$$

即主支路初始权重为 1（可继续生长），次级支路从 0 开始积累。[^src-schrodinger-bridges-generative-modeling]

## 目标函数

目标函数为各支路加权控制成本的积分：

$$\inf_{\{u_k\},\{g_k\}} \sum_{k=0}^K \mathbb{E}_{p_{t,k}}\left[\int_0^T \left(\frac{1}{2}\|u_k(X_t,t)\|^2 + c(X_t,t)\right) w_{t,k} dt\right]$$

终端时刻的支路权重满足 $w_{T,k} = w_{T,k}^\star$（给定的目标分支比例）。[^src-schrodinger-bridges-generative-modeling]

> [!note] 关键洞察
> 每个 mode 由**其自身的 control drift** $u_k$ 独立生成，而非通过单一 control 同时覆盖所有 mode。这使得 branched SB 天然避免 classical SB 在多模态目标下的 mode collapse 问题——如果强行用一个 control 覆盖多个 mode，最优解会取加权平均，导致生成样本落入 mode 之间的低密度区域。

## 主支路与次级支路的配合

分叉结构提供了一种自然的时序建模：

1. **主支路**：所有轨迹共享前段路径（如在细胞分化中，多能干细胞阶段的发育是共享的）
2. **次级支路**：在特定分支点通过 $g_k$ 分流，各自驶向不同的目标 mode
3. **局部控制**：每个支路的 $u_k$ 只需优化该 mode 对应的局部几何，无需协调不相关的 mode

## 与其他 SB 变体的关系

| 变体 | 多目标支持 | 质量变化 | 分支结构 |
|------|----------|---------|---------|
| Classical SB | 弱（单 control） | 否 | 无 |
| [[multi-marginal-schrodinger-bridge\|多边际 SB]] | 通过时间点间接支持 | 否 | 无 |
| [[unbalanced-schrodinger-bridge\|非平衡 SB]] | 弱 | 是 | 无 |
| Branched SB | 强（多 control） | 是 | 有 |

## 应用

- **多峰细胞分化（Multimodal cell differentiation）**：从多能干细胞到多个不同终端细胞类型的发育过程 [^src-schrodinger-bridges-generative-modeling]
- **扰动响应（Perturbation response）**：药物处理后不同细胞亚群走向不同命运
- **生成建模中的多模态目标**：避免 GAN 式 mode collapse 的 principled 框架

## 相关页面

- [[schrodinger-bridge]] — 经典 SB 核心理论
- [[multi-marginal-schrodinger-bridge]] — 多时间点扩展
- [[unbalanced-schrodinger-bridge]] — 非平衡扩展

[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]
