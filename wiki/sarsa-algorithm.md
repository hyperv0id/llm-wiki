---
title: "Sarsa 算法"
type: technique
tags:
  - reinforcement-learning
  - td-learning
  - on-policy
created: 2026-04-27
last_updated: 2026-04-27
source_count: 1
confidence: high
status: active
---

# Sarsa 算法

Sarsa 是 on-policy TD 控制算法，更新依赖五元组 $(s,a,r,s',a')$。[^src-chapter-7-temporal-difference-methods]

标准更新为：$q(s,a)\leftarrow q(s,a)+\alpha[r+\gamma q(s',a')-q(s,a)]$。[^src-chapter-7-temporal-difference-methods]

与 [[q-learning-algorithm|Q-learning]] 相比，Sarsa 评估的是当前行为策略而非直接逼近最优贪心目标。[^src-chapter-7-temporal-difference-methods]

[^src-chapter-7-temporal-difference-methods]: [[source-chapter-7-temporal-difference-methods]]
