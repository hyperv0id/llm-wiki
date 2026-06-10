---
title: "演化道路网络交通预测"
type: concept
tags:
  - spatiotemporal
  - traffic-forecasting
  - continual-learning
  - graph-evolution
created: 2026-06-10
last_updated: 2026-06-10
source_count: 1
confidence: medium
status: active
---

# 演化道路网络交通预测（Evolving RN Traffic Forecasting）

演化道路网络交通预测是由 [[team|TEAM]]（PVLDB 2024）首次形式化的问题：在道路网络拓扑随时间变化（节点/边的增删）的场景下，持续进行交通预测而无需每次全量重训模型[^src-team]。

## 问题形式化

将长期 RN 建模为一系列 graph snapshots {ă₁, ă₂, ..., ăₙ}，其中 ă_τ = ă_(τ−1) + Δă_τ。每个 Δă_τ 包含新增和移除的节点、边以及受影响的已有节点。目标是构建一组函数 {F₁(·), F₂(·), ..., Fₙ(·)}，其中 F_τ 从 F_(τ−1) 迁移而来，仅使用 Δă_τ 对应数据训练[^src-team]。

## 与相关问题的区别

| 问题 | 核心挑战 | 代表工作 |
|------|---------|---------|
| 演化 RN 预测 | 拓扑增删 + 模式漂移 | TEAM (PVLDB 2024) |
| CSTF (扩展图) | 仅节点增加 + 灾难性遗忘 | TrafficStream, STBP |
| 动态图嵌入 | 学习时变节点表示 | EvolveGCN, DyRep |
| 增量学习（分类） | 新类别 + 遗忘 | rehearsal-based CL |

TEAM 的独特之处在于同时处理 **RN 的扩展和收缩**，以及 **模式漂移**（通过 EMD-based 缓冲策略区分稳定/不稳定节点）[^src-team]。

## TEAM 的解决方案

1. **Wasserstein 稳定性度量**：通过 EMD 比较节点在演化前后的数据分布（直方图），量化节点受拓扑变化的影响程度
2. **双缓冲机制**：巩固缓冲 Bc（最稳定节点，用于排练）和更新缓冲 Bu（最不稳定节点，强制重学）
3. **弹性权重巩固**（EWC）：保护对历史任务重要的参数，仅更新次要参数
4. **增量数据训练**：仅用新节点 + 缓冲区数据，复杂度从 O(N²) 降至 O((ΔN+|B|)²)

[^src-team]: [[source-team]]
