---
title: "Chapter 3: Optimal State Values and Bellman Optimality Equation"
type: source-summary
tags:
  - reinforcement-learning
  - bellman-optimality-equation
  - optimal-policy
  - dynamic-programming
created: 2026-04-27
last_updated: 2026-04-27
source_count: 1
confidence: medium
status: active
---

# Chapter 3: Optimal State Values and Bellman Optimality Equation

本章从“策略评估”推进到“策略最优性”，给出最优状态值 $v_*$、最优动作值 $q_*$ 与最优策略的统一定义，并引入 [[bellman-optimality-equation|贝尔曼最优方程]] 作为核心工具。[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]

关键结论是：最优值函数由一个带 max 运算的不动点方程决定，方程右侧映射是压缩映射，因此存在唯一不动点，且从任意初值迭代都可收敛到 $v_*$。该结论把“最优控制”转化为“求不动点”的问题。[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]

本章还给出最优策略结构：虽然最优策略不一定唯一，但总存在确定性贪心最优策略，可通过对 $q_*$ 逐状态取最大动作构造。章节通过网格世界示例展示折扣因子 $\gamma$ 对策略远见性的影响，并讨论奖励仿射变换下最优策略不变性。[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]

方法意义在于：第 4 章的值迭代与策略迭代，实质上都是本章方程与最优性条件的算法化实现。[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]

[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]: [[source-chapter-3-optimal-state-values-and-bellman-optimality-equation]]
