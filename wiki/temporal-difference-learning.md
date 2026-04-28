---
title: "Temporal-Difference Learning"
type: technique
tags:
  - reinforcement-learning
  - td-learning
  - sarsa
  - q-learning
created: 2026-04-27
last_updated: 2026-04-27
source_count: 3
confidence: high
status: active
---

# Temporal-Difference Learning

时序差分学习（Temporal-Difference Learning，TD Learning）是强化学习中的一类核心算法，结合了蒙特卡洛方法和动态规划的思想。它是《[[math-foundation-of-reinforcement-learning|Mathematical Foundations of Reinforcement Learning]]》一书第 7 章的主题。[^src-math-foundation-rl-readme]

## 核心思想

TD 学习不需要等待完整的 episode 结束即可更新值函数估计，而是利用当前奖励和下一状态的估计值进行"自举"（bootstrapping）更新。这使得 TD 方法在在线学习和持续环境中具有显著优势。[^src-math-foundation-rl-readme]

## 主要算法

### Sarsa

Sarsa 是一种 on-policy 的 TD 控制算法，使用当前策略产生的经验元组 $(s, a, r, s', a')$ 来更新动作值函数。算法名称来源于这五个元素的缩写。[^src-math-foundation-rl-readme]

### Expected Sarsa 与 n-step Sarsa

Expected Sarsa 是 Sarsa 的变体，使用动作值的期望而非单个采样值进行更新，通常能降低方差。n-step Sarsa 则结合了多个时间步的信息，在偏差和方差之间取得平衡。[^src-math-foundation-rl-readme]

### Q-learning

Q-learning 是一种 off-policy 的 TD 控制算法，直接学习最优动作值函数 $q_*$，而与智能体实际采取的策略无关。Q-learning 是强化学习中最具影响力的算法之一。[^src-math-foundation-rl-readme]

## 统一视角

该书第 7 章还提供了一个统一的视角，将 Sarsa、Expected Sarsa、n-step Sarsa 和 Q-learning 纳入同一个理论框架下进行比较和分析。[^src-math-foundation-rl-readme]

## 后续发展

TD 学习的思想进一步延伸到了值函数近似（第 8 章，包括 DQN 和经验回放）以及 [[bellman-equation|贝尔曼方程]] 的更广泛应用中。[^src-math-foundation-rl-readme]

## Chapter 7 深化

TD 更新的核心是“目标 - 当前估计”：以状态值为例，目标可写为 $r_{t+1}+\gamma V(S_{t+1})$，仅更新访问到的状态。[^src-chapter-7-temporal-difference-methods]

控制算法中，[[sarsa-algorithm|Sarsa]] 是 on-policy；[[expected-sarsa|Expected Sarsa]] 用期望降低方差；[[n-step-sarsa|n-step Sarsa]] 通过多步目标折中偏差与方差；[[q-learning-algorithm|Q-learning]] 通过 $\max$ 目标实现 off-policy 最优学习。[^src-chapter-7-temporal-difference-methods]

从统一视角看，上述方法都可写成“当前估计 + 学习率 ×（目标 - 当前估计）”框架，仅目标构造不同。[^src-chapter-7-temporal-difference-methods]

## 与 Monte Carlo 的关系

[[monte-carlo-methods-rl|蒙特卡洛方法]] 通常在回合结束后基于完整回报更新，偏差更低但方差较高；TD 方法通过自举更新可更早学习、样本利用更连续。[^src-chapter-5-monte-carlo-methods]

[^src-math-foundation-rl-readme]: [[source-math-foundation-rl-readme]]
[^src-chapter-7-temporal-difference-methods]: [[source-chapter-7-temporal-difference-methods]]
[^src-chapter-5-monte-carlo-methods]: [[source-chapter-5-monte-carlo-methods]]
