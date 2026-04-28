---
title: "Q-learning 算法"
type: technique
tags:
  - reinforcement-learning
  - td-learning
  - off-policy
created: 2026-04-27
last_updated: 2026-04-27
source_count: 1
confidence: high
status: active
---

# Q-learning 算法

Q-learning 是 off-policy TD 控制算法，直接逼近最优动作值函数。[^src-chapter-7-temporal-difference-methods]

其更新为：$q(s,a)\leftarrow q(s,a)+\alpha[r+\gamma\max_{a'}q(s',a')-q(s,a)]$。[^src-chapter-7-temporal-difference-methods]

关键特征是行为策略与目标策略可分离，因此可复用其他策略产生的数据。[^src-chapter-7-temporal-difference-methods]

[^src-chapter-7-temporal-difference-methods]: [[source-chapter-7-temporal-difference-methods]]
