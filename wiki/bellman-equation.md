---
title: "Bellman Equation"
type: technique
tags:
  - reinforcement-learning
  - dynamic-programming
  - value-function
created: 2026-04-27
last_updated: 2026-04-27
source_count: 4
confidence: high
status: active
---

# Bellman Equation

贝尔曼方程（Bellman Equation）是强化学习中的核心理论基础，描述了状态值函数（state value）和动作值函数（action value）之间的递归关系。它是《[[math-foundation-of-reinforcement-learning|Mathematical Foundations of Reinforcement Learning]]》一书第 2 章的核心内容。[^src-math-foundation-rl-readme]

## 基本概念

### 状态值（State Value）

状态值 $v_\pi(s)$ 表示在策略 $\pi$ 下，从状态 $s$ 出发所能获得的期望回报（expected return）。贝尔曼方程将当前状态的值与下一状态的值联系起来，形成递归关系。[^src-math-foundation-rl-readme]

### 动作值（Action Value）

动作值 $q_\pi(s, a)$ 表示在策略 $\pi$ 下，从状态 $s$ 采取动作 $a$ 后所能获得的期望回报。动作值与状态值之间存在紧密的数学联系。[^src-math-foundation-rl-readme]

## 贝尔曼最优方程（Bellman Optimality Equation）

第 3 章进一步介绍了贝尔曼最优方程，用于寻找最优策略。最优策略可以通过求解贝尔曼最优方程得到，该方程描述了最优状态值函数和最优动作值函数之间的关系。[^src-math-foundation-rl-readme]

第 3 章还给出最优性算子的压缩映射性质，并据此保证最优方程存在唯一不动点。对应内容可见 [[bellman-optimality-equation|贝尔曼最优方程]] 与 [[contraction-mapping-theorem|压缩映射定理]]。[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]

## 求解方法

贝尔曼方程可以写成矩阵-向量形式，从而可以通过线性代数方法直接求解。对于大规模问题，则使用迭代方法（如值迭代和策略迭代）进行近似求解。[^src-math-foundation-rl-readme]

## Chapter 2 深化

在给定策略 $\pi$ 时，状态值满足通式：$v_\pi(s)=\sum_a\pi(a|s)[\sum_r p(r|s,a)r+\gamma\sum_{s'}p(s'|s,a)v_\pi(s')]$，对应的矩阵形式为 $v_\pi=r_\pi+\gamma P_\pi v_\pi$。[^src-chapter-2-state-values-and-bellman-equation]

当 $\gamma\in(0,1)$ 且 $P_\pi$ 为随机矩阵时，可写出闭式解 $v_\pi=(I-\gamma P_\pi)^{-1}r_\pi$，也可用迭代策略评估逼近不动点。[^src-chapter-2-state-values-and-bellman-equation]

## 在本书中的位置

贝尔曼方程是整本书的数学基石。后续章节中的 [[temporal-difference-learning|时序差分学习]]、Q-learning、策略梯度等方法都建立在贝尔曼方程的理论基础之上。[^src-math-foundation-rl-readme]

第 4 章的 [[value-iteration|值迭代]]、[[policy-iteration|策略迭代]] 与 [[truncated-policy-iteration|截断策略迭代]] 都可视作该方程及其最优形式的数值求解流程。[^src-chapter-4-value-iteration-and-policy-iteration]

[^src-math-foundation-rl-readme]: [[source-math-foundation-rl-readme]]
[^src-chapter-2-state-values-and-bellman-equation]: [[source-chapter-2-state-values-and-bellman-equation]]
[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]: [[source-chapter-3-optimal-state-values-and-bellman-optimality-equation]]
[^src-chapter-4-value-iteration-and-policy-iteration]: [[source-chapter-4-value-iteration-and-policy-iteration]]
