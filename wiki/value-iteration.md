---
title: "值迭代"
type: technique
tags:
  - reinforcement-learning
  - dynamic-programming
  - value-iteration
created: 2026-04-27
last_updated: 2026-04-27
source_count: 2
confidence: high
status: active
---

# 值迭代

值迭代是直接求解 [[bellman-optimality-equation|贝尔曼最优方程]] 的动态规划算法。[^src-chapter-4-value-iteration-and-policy-iteration]

每轮迭代先执行贪心改进，再更新值函数；反复执行后，值函数收敛到最优值函数。[^src-chapter-4-value-iteration-and-policy-iteration]

该算法建立在最优性算子的压缩映射性质之上，因此从任意初值出发都可收敛。[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]

[^src-chapter-4-value-iteration-and-policy-iteration]: [[source-chapter-4-value-iteration-and-policy-iteration]]
[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]: [[source-chapter-3-optimal-state-values-and-bellman-optimality-equation]]
