---
title: "ε-greedy 策略"
type: technique
tags:
  - reinforcement-learning
  - exploration
  - exploitation
  - epsilon-greedy
created: 2026-04-27
last_updated: 2026-04-27
source_count: 1
confidence: high
status: active
---

# ε-greedy 策略

ε-greedy 用于平衡探索与利用：以较高概率选择当前贪心动作，以较小概率随机探索其他动作。[^src-chapter-5-monte-carlo-methods]

该机制使无模型算法在不依赖 Exploring Starts 的情况下仍能覆盖动作空间。[^src-chapter-5-monte-carlo-methods]

它是 [[monte-carlo-methods-rl|MC 方法]]、[[sarsa-algorithm|Sarsa]] 与 [[q-learning-algorithm|Q-learning]] 的常见行为策略。[^src-chapter-5-monte-carlo-methods]

[^src-chapter-5-monte-carlo-methods]: [[source-chapter-5-monte-carlo-methods]]
