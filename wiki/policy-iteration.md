---
title: "策略迭代"
type: technique
tags:
  - reinforcement-learning
  - dynamic-programming
  - policy-iteration
created: 2026-04-27
last_updated: 2026-04-27
source_count: 2
confidence: high
status: active
---

# 策略迭代

策略迭代由两步交替组成：[[policy-evaluation|策略评估]] 与策略改进。[^src-chapter-4-value-iteration-and-policy-iteration]

在每轮中，先评估当前策略的值函数，再按贪心准则改进策略；根据策略改进引理，策略性能单调不降并收敛到最优策略。[^src-chapter-4-value-iteration-and-policy-iteration]

它与 [[value-iteration|值迭代]] 的区别在于：每轮会做更充分的评估，通常轮数更少但单轮成本更高。[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]

[^src-chapter-4-value-iteration-and-policy-iteration]: [[source-chapter-4-value-iteration-and-policy-iteration]]
[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]: [[source-chapter-3-optimal-state-values-and-bellman-optimality-equation]]
