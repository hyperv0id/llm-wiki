---
title: "Mathematical Foundations of Reinforcement Learning"
type: entity
tags:
  - reinforcement-learning
  - textbook
  - springer
  - tsinghua-university-press
created: 2026-04-27
last_updated: 2026-04-27
source_count: 7
confidence: high
status: active
---

# Mathematical Foundations of Reinforcement Learning

《Mathematical Foundations of Reinforcement Learning》是 [[shiyu-zhao|Shiyu Zhao]] 撰写的一部强化学习教材，由 Springer Press 和清华大学出版社联合出版（2025）。该书旨在提供**数学化但友好**的强化学习基础介绍。[^src-math-foundation-rl-readme]

## 内容概述

全书共十章，分为"基本工具"和"算法"两大部分，各章高度关联，建议顺序学习。书中所有示例均基于 [[grid-world-environment|网格世界]] 任务。[^src-math-foundation-rl-readme]

涵盖主题：
- 基本概念：状态、动作、策略、奖励、回报、马尔可夫决策过程（MDP）
- [[bellman-equation|贝尔曼方程]]与贝尔曼最优方程
- 值迭代与策略迭代
- 蒙特卡洛学习
- 随机近似与随机梯度下降（SGD）
- [[temporal-difference-learning|时序差分学习]]（Sarsa、Expected Sarsa、n-step Sarsa、Q-learning）
- 值函数近似与 DQN（经验回放）
- 策略梯度方法（REINFORCE）
- Actor-Critic 方法（优势 Actor-Critic、重要性采样、确定性 Actor-Critic）[^src-math-foundation-rl-readme]

深度 ingest 后，可把主线聚焦为三章：第 1 章（基础建模，见 [[mdp-formal-definition|MDP 形式化定义]]）、第 2 章（值函数与 [[bellman-equation|贝尔曼方程]]，见 [[policy-evaluation|策略评估]] 与 [[action-value-function|动作值函数]]）、第 7 章（[[temporal-difference-learning|TD 学习]]，见 [[sarsa-algorithm|Sarsa]]、[[expected-sarsa|Expected Sarsa]]、[[n-step-sarsa|n-step Sarsa]]、[[q-learning-algorithm|Q-learning]]）。[^src-chapter-1-basic-concepts][^src-chapter-2-state-values-and-bellman-equation][^src-chapter-7-temporal-difference-methods]

进一步扩展后，第 3/4/5 章形成“最优性方程 → 动态规划算法 → 无模型采样方法”的连续链路：[[bellman-optimality-equation|贝尔曼最优方程]]、[[value-iteration|值迭代]]、[[policy-iteration|策略迭代]]、[[truncated-policy-iteration|截断策略迭代]]、[[monte-carlo-methods-rl|蒙特卡洛方法]] 与 [[epsilon-greedy|ε-greedy]]。[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation][^src-chapter-4-value-iteration-and-policy-iteration][^src-chapter-5-monte-carlo-methods]

## 配套资源

- 中文授课视频（Bilibili）和英文授课视频（YouTube），累计 **2,100,000+ 次观看**
- 官方 [[grid-world-environment|网格世界环境]] 代码（Python + MATLAB）
- 第三方社区贡献的多语言代码实现（Python、MATLAB、R、C++）[^src-math-foundation-rl-readme]

## 读者反馈

截至 2025 年 6 月，该仓库在 GitHub 上获得 **10,000+ stars**，收到大量正面评价。读者来自全球，包括学生、研究人员和从业者。[^src-math-foundation-rl-readme]

[^src-math-foundation-rl-readme]: [[source-math-foundation-rl-readme]]
[^src-chapter-1-basic-concepts]: [[source-chapter-1-basic-concepts]]
[^src-chapter-2-state-values-and-bellman-equation]: [[source-chapter-2-state-values-and-bellman-equation]]
[^src-chapter-7-temporal-difference-methods]: [[source-chapter-7-temporal-difference-methods]]
[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]: [[source-chapter-3-optimal-state-values-and-bellman-optimality-equation]]
[^src-chapter-4-value-iteration-and-policy-iteration]: [[source-chapter-4-value-iteration-and-policy-iteration]]
[^src-chapter-5-monte-carlo-methods]: [[source-chapter-5-monte-carlo-methods]]
