---
title: "Fine-grained Traffic Prediction"
type: concept
tags:
  - traffic-forecasting
  - lane-level
  - fine-grained
  - multi-granularity
created: 2026-07-16
last_updated: 2026-07-16
source_count: 1
confidence: medium
status: active
---

# Fine-grained Traffic Prediction

**细粒度交通预测**（Fine-grained Traffic Prediction）指同时涵盖道路级（road-level）和车道级（lane-level）的交通状态预测任务，相较于传统大规模城市交通预测提供更精细的数据粒度，对精准城市管理、自动驾驶车道变更引导和动态车道系统至关重要[^src-minitraffic]。

## 问题形式化

[[minitraffic|MiniTraffic]] 将细粒度交通网络定义为两个独立的无向图[^src-minitraffic]：

- **道路网络**：$\mathcal{G}^R = (V^R, E^R, A^R)$，节点 $r_i$ 表示第 $i$ 个道路段，$N^R = I$
- **车道网络**：$\mathcal{G}^L = (V^L, E^L, A^L)$，节点 $l_{i,j}$ 表示第 $i$ 个道路段的第 $j$ 条车道，$N^L = \sum_{i=1}^I J_i$

核心是领域迁移学习：从多源道路域数据 $\mathcal{G}^{Source}$ 预训练，迁移到仅有少量标注的车道域 $\mathcal{G}^L$[^src-minitraffic]。

## 关键挑战

1. **数据不平衡**：道路级数据丰富（如 METR-LA 650 万观测点），车道级数据稀缺且获取成本高（传感器部署+人工标注）[^src-minitraffic]。
2. **多粒度关联**：道路和车道交通相互影响，需同时建模两级粒度以利用关联信息[^src-minitraffic]。
3. **计算资源约束**：细粒度场景数据变化快，需频繁重训/微调，大型模型（如 [[urbangpt|UrbanGPT]]、[[unist|UniST]]）部署成本不可接受[^src-minitraffic]。

## 与相关概念的关系

- vs **大规模城市交通预测**（如 [[unist|UniST]]、[[urbandit|UrbanDiT]]）：细粒度预测关注路段内部的多车道动态，而非区域级别的流量统计[^src-minitraffic]。
- vs **道路级预测**（如 [[dcrnn|DCRNN]]、[[gwnet|GWNet]]）：道路级预测是细粒度预测的子任务；车道级预测是更精细的层次[^src-minitraffic]。
- 与 [[mcgvae|McgVAE]] 的关系：McgVAE（CIKM 2024）是首个尝试道路-车道联合建模的方法，但采用集成架构，无预训练机制[^src-minitraffic]。

[^src-minitraffic]: [[source-minitraffic]]
