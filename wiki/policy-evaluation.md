---
title: "策略评估"
type: technique
tags:
  - reinforcement-learning
  - bellman-equation
  - value-function
created: 2026-04-27
last_updated: 2026-04-27
source_count: 2
confidence: high
status: active
---

# 策略评估

策略评估指在给定策略 $\pi$ 时求解其状态值函数 $v_\pi$ 的过程。[^src-chapter-2-state-values-and-bellman-equation]

其核心方程是 [[bellman-equation|贝尔曼方程]]，矩阵形式为 $v_\pi=r_\pi+\gamma P_\pi v_\pi$，可用闭式解或迭代法求解。[^src-chapter-2-state-values-and-bellman-equation]

该过程是策略迭代、TD 学习与价值方法的公共子程序。[^src-chapter-2-state-values-and-bellman-equation]

在 [[policy-iteration|策略迭代]] 中，策略评估与策略改进交替执行；在 [[truncated-policy-iteration|截断策略迭代]] 中，每轮仅执行有限步评估以折中计算量与收敛速度。[^src-chapter-4-value-iteration-and-policy-iteration]

[^src-chapter-2-state-values-and-bellman-equation]: [[source-chapter-2-state-values-and-bellman-equation]]
[^src-chapter-4-value-iteration-and-policy-iteration]: [[source-chapter-4-value-iteration-and-policy-iteration]]
