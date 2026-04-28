---
title: "策略评估"
type: technique
tags:
  - reinforcement-learning
  - bellman-equation
  - value-function
created: 2026-04-27
last_updated: 2026-04-28
source_count: 2
confidence: high
status: active
---

# 策略评估

策略评估是在给定策略 $\pi$ 时求解其状态值函数 $v_\pi$ 的过程。它是所有值方法与策略改进算法的公共子程序。[^src-chapter-2-state-values-and-bellman-equation]

## 问题形式化

在已知环境模型（转移概率 $P$ 和奖励函数 $R$）的情况下，策略评估等价于求解 [[bellman-equation|贝尔曼方程]]：

$$v_\pi(s) = \sum_a \pi(a|s) \left[ \sum_r p(r|s,a) r + \gamma \sum_{s'} p(s'|s,a) v_\pi(s') \right]$$

## 求解方法

### 闭式解

对于有限状态问题，贝尔曼方程可写成矩阵-向量形式：

$$v_\pi = r_\pi + \gamma P_\pi v_\pi$$

当 $\gamma \in (0,1)$ 且 $P_\pi$ 为随机矩阵时，可直接求解：

$$v_\pi = (I - \gamma P_\pi)^{-1} r_\pi$$

但在大状态空间中，矩阵求逆的计算复杂度为 $O(|\mathcal{S}|^3)$，代价较高。[^src-chapter-2-state-values-and-bellman-equation]

### 迭代策略评估

实践中常使用迭代法逼近 $v_\pi$：

$$v_{k+1}(s) = \sum_a \pi(a|s) \left[ \sum_r p(r|s,a) r + \gamma \sum_{s'} p(s'|s,a) v_k(s') \right]$$

从任意初始 $v_0$ 出发，由于 $P_\pi$ 是随机矩阵且 $\gamma < 1$，迭代收敛到不动点 $v_\pi$。收敛条件为 $\max_s |v_{k+1}(s) - v_k(s)| < \theta$（$\theta$ 为收敛阈值）。

## 在算法中的作用

策略评估在以下算法中被调用：[^src-chapter-4-value-iteration-and-policy-iteration]

- **[[policy-iteration|策略迭代]]**：每轮执行完整策略评估，再执行策略改进
- **[[truncated-policy-iteration|截断策略迭代]]**：每轮仅执行有限步评估，折中计算量与收敛速度
- **广义策略迭代（GPI）**：评估和改进交错进行、异步执行

策略评估是连接 [[bellman-equation|贝尔曼方程]] 理论与实际算法的桥梁。[^src-chapter-2-state-values-and-bellman-equation]

[^src-chapter-2-state-values-and-bellman-equation]: [[source-chapter-2-state-values-and-bellman-equation]]
[^src-chapter-4-value-iteration-and-policy-iteration]: [[source-chapter-4-value-iteration-and-policy-iteration]]
