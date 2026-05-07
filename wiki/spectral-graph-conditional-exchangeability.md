---
title: "Spectral Graph Conditional Exchangeability (SGCE)"
type: concept
tags:
  - conformal-prediction
  - spectral-graph-theory
  - exchangeability
  - graph-wavelet
created: 2026-05-07
last_updated: 2026-05-07
source_count: 1
confidence: medium
status: active
---

# Spectral Graph Conditional Exchangeability (SGCE)

**谱图条件可交换性**是 Guo et al.（ICML 2026）为图结构多变量时间序列共形预测提出的概念。[^src-scale]

## 核心思想

图结构 MTS 中的跨节点耦合破坏了联合可交换性。根据谱图理论，这种耦合主要体现在全局趋势（低频分量）中，而高频分量（局部变化）的跨节点交互较弱，更接近可交换。

形式定义：残差过程满足 SGCE，当且仅当对任意时间索引序列 $\{t_1, \ldots, t_n\}$ 和任意置换 $\sigma$，有：
$$p(H_{t_1}, \ldots, H_{t_n} | L_{t_1}, \ldots, L_{t_n}) = p(H_{t_{\sigma(1)}}, \ldots, H_{t_{\sigma(n)}} | L_{t_{\sigma(1)}}, \ldots, L_{t_{\sigma(n)}})$$

其中 $H_t$ 是高频分量，$L_t$ 是低频分量。[^src-scale]

## 实证验证

对 METR-LA 数据集的相关系数强度分析显示：
- 低频分量：节点间相关系数高，连接密集（强耦合）→ 不可交换
- 高频分量：节点间相关系数低，连接稀疏（弱耦合）→ 近似可交换
- 两种分量能量水平相当，但耦合强度显著不同[^src-scale]

## 意义

SGCE 为图结构时间序列的共形预测提供了新的理论基础：无需在原始空间处理复杂的跨节点耦合，而是通过谱分解将不可交换的低频信息作为条件，对可交换的高频分量进行共形校准。这使得在保持全局趋势的同时实现有效的预测区间。[^src-scale]

## 关联方法

- [[scale]] — SGCE 的算法实现
- [[conformal-prediction]] — 共形预测基础
- [[spectral-graph-wavelet-transform]] — 谱图小波变换

[^src-scale]: [[source-scale]]
