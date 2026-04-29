---
title: "Chapter 5: Monte Carlo Methods"
type: source-summary
tags:
  - reinforcement-learning
  - monte-carlo
  - epsilon-greedy
  - exploration
created: 2026-04-27
last_updated: 2026-04-27
source_count: 1
confidence: medium
status: active
---

# Chapter 5: Monte Carlo Methods

本章进入无模型强化学习，核心思想是：在未知转移与奖励模型时，依赖采样轨迹估计值函数，并通过策略改进逼近最优策略。[^src-chapter-5-monte-carlo-methods]

章节先用均值估计与大数定律建立统计直觉，再给出 MC Basic、MC Exploring Starts 与 MC $\varepsilon$-Greedy 三类算法。MC Basic 结构清晰但样本效率低；Exploring Starts 通过增强覆盖提升利用率，但在真实系统里往往难满足起始-动作可控假设。[^src-chapter-5-monte-carlo-methods]

为摆脱 Exploring Starts 假设，本章引入 [[epsilon-greedy|$\varepsilon$-greedy]] 政策改进：以高概率执行当前最优动作、以小概率探索其他动作，从而在利用与探索之间形成可调平衡。该机制成为后续 TD 与深度 RL 的常见默认探索策略。[^src-chapter-5-monte-carlo-methods]

本章还强调稀疏奖励与回合长度对 MC 估计稳定性的影响，说明算法性能不仅由更新公式决定，还依赖任务设计与数据分布。[^src-chapter-5-monte-carlo-methods]

[^src-chapter-5-monte-carlo-methods]: [[source-chapter-5-monte-carlo-methods]]
