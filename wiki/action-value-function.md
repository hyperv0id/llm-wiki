---
title: "动作值函数"
type: concept
tags:
  - reinforcement-learning
  - q-function
  - value-function
created: 2026-04-27
last_updated: 2026-04-27
source_count: 2
confidence: high
status: active
---

# 动作值函数

动作值函数 $q_\pi(s,a)$ 表示在策略 $\pi$ 下，从状态 $s$ 执行动作 $a$ 后的期望回报。[^src-chapter-2-state-values-and-bellman-equation]

它与状态值满足关系 $v_\pi(s)=\sum_a\pi(a|s)q_\pi(s,a)$，是从“评估状态”转向“评估动作”的关键桥梁。[^src-chapter-2-state-values-and-bellman-equation]

[[sarsa-algorithm|Sarsa]]、[[expected-sarsa|Expected Sarsa]]、[[n-step-sarsa|n-step Sarsa]] 和 [[q-learning-algorithm|Q-learning]] 都直接更新 $q$ 函数估计。[^src-chapter-7-temporal-difference-methods]

[^src-chapter-2-state-values-and-bellman-equation]: [[source-chapter-2-state-values-and-bellman-equation]]
[^src-chapter-7-temporal-difference-methods]: [[source-chapter-7-temporal-difference-methods]]
