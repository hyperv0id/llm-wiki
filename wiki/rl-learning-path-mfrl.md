---
title: "RL 学习路径：基于 Mathematical Foundations of Reinforcement Learning"
type: analysis
tags:
  - reinforcement-learning
  - learning-path
  - textbook-guide
created: 2026-04-27
last_updated: 2026-04-27
source_count: 8
confidence: medium
status: active
---

# RL 学习路径：基于 Mathematical Foundations of Reinforcement Learning

本文分析如何利用《[[math-foundation-of-reinforcement-learning|Mathematical Foundations of Reinforcement Learning]]》及其配套 [[grid-world-environment|网格世界环境]] 代码系统性地学习强化学习。[^src-math-foundation-rl-readme][^src-grid-world-code-readme]

## 学习路径概览

该书采用递进式结构，每章建立在前一章的基础上。以下是根据该书内容设计的学习路径：

### 第一阶段：基础概念（第 1 章）

从最基本的概念开始：状态、动作、策略、奖励、回报和马尔可夫决策过程（MDP）。这是理解后续所有内容的必要前提。[^src-math-foundation-rl-readme]

建议在该阶段同步阅读 [[mdp-formal-definition|MDP 形式化定义]]，并用 [[grid-world-environment|网格世界环境]] 验证状态转移与奖励设计。[^src-chapter-1-basic-concepts]

### 第二阶段：数学基础（第 2–3 章）

深入学习 [[bellman-equation|贝尔曼方程]] 和贝尔曼最优方程。这些方程是整个强化学习理论的数学基石。建议配合矩阵-向量形式的推导进行理解。[^src-math-foundation-rl-readme]

这一阶段可并行学习 [[policy-evaluation|策略评估]] 与 [[action-value-function|动作值函数]]，建立“状态值—动作值—策略”闭环。[^src-chapter-2-state-values-and-bellman-equation]

第 3 章建议重点攻克 [[bellman-optimality-equation|贝尔曼最优方程]] 与 [[contraction-mapping-theorem|压缩映射定理]]，理解“最优控制 = 不动点求解”的视角。[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]

### 第三阶段：经典算法（第 4–5 章）

学习值迭代、策略迭代和蒙特卡洛方法。这些算法展示了如何利用贝尔曼方程进行策略评估和改进。蒙特卡洛方法引入了基于采样（sampling）的思想。[^src-math-foundation-rl-readme]

建议按“[[value-iteration|值迭代]] → [[policy-iteration|策略迭代]] → [[truncated-policy-iteration|截断策略迭代]]”比较评估成本与收敛速度，再进入 [[monte-carlo-methods-rl|蒙特卡洛方法]] 与 [[epsilon-greedy|ε-greedy]]。[^src-chapter-4-value-iteration-and-policy-iteration][^src-chapter-5-monte-carlo-methods]

这一阶段可专门复盘 [[exploration-vs-exploitation|探索与利用]] 问题，为第 7 章 TD 方法打基础。[^src-chapter-5-monte-carlo-methods]

### 第四阶段：随机优化（第 6 章）

随机近似和随机梯度下降（SGD）为后续的 TD 学习提供了数学基础。理解 Robbins-Monro 算法和 SGD 的收敛性质对理解 TD 算法至关重要。[^src-math-foundation-rl-readme]

### 第五阶段：TD 学习（第 7 章）

[[temporal-difference-learning|时序差分学习]] 是本书的核心算法章节，涵盖 Sarsa、Expected Sarsa、n-step Sarsa 和 Q-learning。建议在 [[grid-world-environment|网格世界环境]] 中动手实现这些算法。[^src-math-foundation-rl-readme][^src-grid-world-code-readme]

建议按顺序实现 [[sarsa-algorithm|Sarsa]] → [[expected-sarsa|Expected Sarsa]] → [[n-step-sarsa|n-step Sarsa]] → [[q-learning-algorithm|Q-learning]]，并结合 [[on-policy-vs-off-policy|On-policy 与 Off-policy]] 做对照实验。[^src-chapter-7-temporal-difference-methods]

### 第六阶段：进阶方法（第 8–10 章）

- **值函数近似与 DQN**：将 TD 学习扩展到大规模状态空间，引入经验回放机制
- **策略梯度方法**：直接优化策略参数（REINFORCE）
- **Actor-Critic 方法**：结合值函数和策略梯度的优势[^src-math-foundation-rl-readme]

## 实践建议

1. 配合授课视频学习效果更佳（中文视频在 Bilibili，英文视频在 YouTube）
2. 使用官方 [[grid-world-environment|网格世界环境]] 代码动手实现各章算法
3. 参考第三方社区贡献的代码实现（Python、MATLAB、R、C++）[^src-math-foundation-rl-readme][^src-grid-world-code-readme]

## 前置知识

- 概率论基础
- 线性代数基础
- 书中附录也包含了所需数学基础的复习材料[^src-math-foundation-rl-readme]

[^src-math-foundation-rl-readme]: [[source-math-foundation-rl-readme]]
[^src-grid-world-code-readme]: [[source-grid-world-code-readme]]
[^src-chapter-1-basic-concepts]: [[source-chapter-1-basic-concepts]]
[^src-chapter-2-state-values-and-bellman-equation]: [[source-chapter-2-state-values-and-bellman-equation]]
[^src-chapter-7-temporal-difference-methods]: [[source-chapter-7-temporal-difference-methods]]
[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]: [[source-chapter-3-optimal-state-values-and-bellman-optimality-equation]]
[^src-chapter-4-value-iteration-and-policy-iteration]: [[source-chapter-4-value-iteration-and-policy-iteration]]
[^src-chapter-5-monte-carlo-methods]: [[source-chapter-5-monte-carlo-methods]]
