---
title: "Mathematical Foundations of Reinforcement Learning (Readme)"
type: source-summary
tags:
  - reinforcement-learning
  - textbook
  - springer
  - grid-world
created: 2026-04-27
last_updated: 2026-04-27
source_count: 1
confidence: high
status: active
---

# Mathematical Foundations of Reinforcement Learning — Readme

该仓库是《Mathematical Foundations of Reinforcement Learning》（Springer Press, 2025）一书的配套主页，作者为 [[shiyu-zhao|Shiyu Zhao]]。该书旨在提供**数学化但友好**的强化学习基础概念、基本问题和经典算法介绍。[^src-math-foundation-rl-readme]

## 核心主张

该书从数学视角引入强化学习，帮助读者不仅知道算法的步骤，还能理解其设计动机和有效性原因。数学深度被控制在适当水平，并以精心设计的方式呈现，读者可根据兴趣选择性阅读灰色框中的内容。[^src-math-foundation-rl-readme]

## 内容结构

全书共十章，分为两部分：第一部分介绍基本工具，第二部分介绍算法。各章高度关联，建议按顺序学习。书中所有示例均基于一个网格世界（grid world）任务，便于理解概念和算法。[^src-math-foundation-rl-readme]

涵盖主题包括：基本概念（状态、动作、策略、奖励、回报、马尔可夫决策过程）、贝尔曼方程、贝尔曼最优方程、值迭代与策略迭代、蒙特卡洛学习、随机近似与 SGD、时序差分学习（Sarsa、Q-learning）、值函数近似（DQN）、策略梯度方法、Actor-Critic 方法。[^src-math-foundation-rl-readme]

## 目标读者

面向高年级本科生、研究生、研究人员和从业者。无需强化学习背景，但需要概率论和线性代数基础。书中附录也包含了所需数学基础。[^src-math-foundation-rl-readme]

## 配套资源

- 中文和英文授课视频（Bilibili / YouTube），累计 **2,100,000+ 次观看**
- 第三方社区贡献的 Python、MATLAB、R、C++ 代码实现
- 网格世界环境的官方代码（Python + MATLAB）[^src-math-foundation-rl-readme]

## 局限性

- 官方仅提供网格世界环境代码，不提供所有算法的完整实现（这些作为课后作业）
- 第三方代码未经作者验证
- 作者因事务繁忙，讨论区回复可能存在显著延迟[^src-math-foundation-rl-readme]

[^src-math-foundation-rl-readme]: [[source-math-foundation-rl-readme]]
