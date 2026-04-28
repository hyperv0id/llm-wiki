---
title: "贝尔曼最优方程"
type: technique
tags:
  - reinforcement-learning
  - bellman-optimality-equation
  - optimal-control
created: 2026-04-27
last_updated: 2026-04-27
source_count: 2
confidence: high
status: active
---

# 贝尔曼最优方程

贝尔曼最优方程（BOE）用于刻画最优值函数，是从 [[bellman-equation|贝尔曼方程]] 过渡到最优控制的关键方程。[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]

其核心形式可写为“对动作（或策略）取最大”的不动点方程；该最优性算子在折扣 MDP 下是压缩映射，因此存在唯一不动点 $v_*$。[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]

由 $v_*$ 可构造贪心最优策略；最优策略可能不唯一，但总存在确定性最优策略。[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]

第 4 章的 [[value-iteration|值迭代]] 与 [[policy-iteration|策略迭代]] 都是在数值上求解或逼近该方程。[^src-chapter-4-value-iteration-and-policy-iteration]

[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]: [[source-chapter-3-optimal-state-values-and-bellman-optimality-equation]]
[^src-chapter-4-value-iteration-and-policy-iteration]: [[source-chapter-4-value-iteration-and-policy-iteration]]
