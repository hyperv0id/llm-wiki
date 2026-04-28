---
title: "n-step Sarsa"
type: technique
tags:
  - reinforcement-learning
  - td-learning
  - multi-step
created: 2026-04-27
last_updated: 2026-04-27
source_count: 1
confidence: high
status: active
---

# n-step Sarsa

n-step Sarsa 将单步 Sarsa 扩展为多步目标更新，在偏差与方差之间折中。[^src-chapter-7-temporal-difference-methods]

当 $n=1$ 时退化为 [[sarsa-algorithm|Sarsa]]，当 $n\to\infty$（回合任务）时逼近蒙特卡洛回报。[^src-chapter-7-temporal-difference-methods]

因此它常被用作连接 TD 与 MC 的统一教学入口。[^src-chapter-7-temporal-difference-methods]

[^src-chapter-7-temporal-difference-methods]: [[source-chapter-7-temporal-difference-methods]]
