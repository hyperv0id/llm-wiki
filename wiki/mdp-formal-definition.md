---
title: "MDP 形式化定义"
type: concept
tags:
  - reinforcement-learning
  - mdp
  - markov-property
created: 2026-04-27
last_updated: 2026-04-27
source_count: 1
confidence: high
status: active
---

# MDP 形式化定义

马尔可夫决策过程（MDP）是强化学习的标准问题模型，用于描述“智能体—环境”交互。[^src-chapter-1-basic-concepts]

有限 MDP 可由状态集合、动作集合、奖励、状态转移概率、奖励分布与折扣因子共同定义；其核心是马尔可夫性质：下一步只依赖当前状态与动作。[^src-chapter-1-basic-concepts]

在固定策略后，MDP 退化为马尔可夫过程，因此“控制”可分解为 [[policy-evaluation|策略评估]] 与策略改进两步。[^src-chapter-1-basic-concepts]

[^src-chapter-1-basic-concepts]: [[source-chapter-1-basic-concepts]]
