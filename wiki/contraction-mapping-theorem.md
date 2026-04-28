---
title: "压缩映射定理"
type: concept
tags:
  - reinforcement-learning
  - fixed-point
  - contraction-mapping
created: 2026-04-27
last_updated: 2026-04-27
source_count: 1
confidence: high
status: active
---

# 压缩映射定理

压缩映射定理说明：若算子在某范数下是压缩的，则存在唯一不动点，且迭代会收敛到该不动点。[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]

在强化学习里，第 3 章用该定理证明最优性算子存在唯一解，从而支撑 [[bellman-optimality-equation|贝尔曼最优方程]] 的可解性与迭代收敛性。[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]

第 4 章的 [[value-iteration|值迭代]] 正是这一不动点迭代理论的直接算法实现。[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]

[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]: [[source-chapter-3-optimal-state-values-and-bellman-optimality-equation]]
