---
title: "蒙特卡洛方法（RL）"
type: technique
tags:
  - reinforcement-learning
  - monte-carlo
  - model-free
created: 2026-04-27
last_updated: 2026-04-27
source_count: 1
confidence: medium
status: active
---

# 蒙特卡洛方法（RL）

蒙特卡洛方法在无模型条件下通过完整轨迹回报估计值函数，并结合策略改进逼近最优策略。[^src-chapter-5-monte-carlo-methods]

第 5 章主要给出 MC Basic、MC Exploring Starts 和 MC $\varepsilon$-Greedy 三类方法，差别在于样本利用率和探索机制。[^src-chapter-5-monte-carlo-methods]

与 [[temporal-difference-learning|TD 学习]] 相比，MC 通常方差更高但偏差更低，且依赖回合终止后回报。[^src-chapter-5-monte-carlo-methods]

[^src-chapter-5-monte-carlo-methods]: [[source-chapter-5-monte-carlo-methods]]
