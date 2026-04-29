---
title: "Chapter 2: State Values and Bellman Equation"
type: source-summary
tags:
  - reinforcement-learning
  - bellman-equation
  - value-function
  - policy-evaluation
created: 2026-04-27
last_updated: 2026-04-27
source_count: 1
confidence: medium
status: active
---

# Chapter 2: State Values and Bellman Equation

本章从“如何评价一个策略”切入，系统引入状态值函数 $v_\pi(s)$ 与动作值函数 $q_\pi(s,a)$，并将二者放入贝尔曼递归框架中统一处理。[^src-chapter-2-state-values-and-bellman-equation]

核心结论是：给定策略时，值函数满足线性递归关系，既可逐状态展开，也可写成矩阵形式 $v_\pi=r_\pi+\gamma P_\pi v_\pi$。该形式直接连接了概率转移、即时奖励和长期回报，是后续动态规划与 TD 学习的共同基础。[^src-chapter-2-state-values-and-bellman-equation]

本章进一步给出两种求解路径：其一是闭式解 $v_\pi=(I-\gamma P_\pi)^{-1}r_\pi$；其二是迭代策略评估 $v_{k+1}=r_\pi+\gamma P_\pi v_k$。这说明“评估”既可用线性代数视角理解，也可用迭代不动点视角理解。[^src-chapter-2-state-values-and-bellman-equation]

在关系层面，本章明确了 $v_\pi(s)=\sum_a\pi(a|s)q_\pi(s,a)$，并给出由 $v_\pi$ 反推 $q_\pi$ 的计算式，形成“状态值—动作值—策略”三者闭环。[^src-chapter-2-state-values-and-bellman-equation]

方法意义是：策略评估成为可重复调用的模块，后续策略迭代、Sarsa、Q-learning 等都可视作在不同目标下重复执行“近似贝尔曼求解”。[^src-chapter-2-state-values-and-bellman-equation]

[^src-chapter-2-state-values-and-bellman-equation]: [[source-chapter-2-state-values-and-bellman-equation]]
