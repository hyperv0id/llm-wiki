---
title: "策略迭代"
type: technique
tags:
  - reinforcement-learning
  - dynamic-programming
  - policy-iteration
created: 2026-04-27
last_updated: 2026-04-28
source_count: 3
confidence: high
status: active
---

# 策略迭代

策略迭代（Policy Iteration）是求解最优策略的动态规划算法，由两步交替组成：[[policy-evaluation|策略评估]] 与策略改进。[^src-chapter-4-value-iteration-and-policy-iteration]

## 算法描述

策略迭代的核心流程：

1. **初始化**：任意初始策略 $\pi_0$
2. **策略评估**：计算当前策略 $\pi_k$ 下的状态值函数 $v_{\pi_k}$，即求解 $v_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}$
3. **策略改进**：对每个状态 $s$，按贪心准则改进策略：$\pi_{k+1}(s) = \arg\max_a \sum_{s'} p(s'|s,a)\left(r(s,a,s') + \gamma v_{\pi_k}(s')\right)$
4. 重复步骤 2-3 直到策略不再改变

## 策略改进引理

策略改进引理（Policy Improvement Theorem）是策略迭代收敛的关键保证：

> 若存在策略 $\pi'$ 使得对所有状态 $s$ 有 $q_\pi(s, \pi'(s)) \geq v_\pi(s)$，则 $\pi'$ 至少和 $\pi$ 一样好。即 $v_{\pi'}(s) \geq v_\pi(s)$ 对所有 $s$ 成立。

由于状态有限，策略也是有限的。每次改进后策略性能单调不降，算法必然在有限轮内收敛到最优策略。[^src-chapter-4-value-iteration-and-policy-iteration]

## 与值迭代的比较

| 特性 | 策略迭代 | [[value-iteration|值迭代]] |
|------|----------|---------------------|
| 核心操作 | 策略评估 + 策略改进 | 值函数更新（贝尔曼最优） |
| 每轮成本 | 较高（需解线性系统） | 较低（一次全扫描） |
| 收敛轮数 | 较少（通常 < 10） | 较多 |
| 策略更新频率 | 每轮更新 | 收敛后提取 |

策略迭代与值迭代是 [[truncated-policy-iteration|截断策略迭代]] 的两个端点。[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]

## 变体

- **广义策略迭代（GPI）**：评估和改进可交错执行、异步进行，不要求评估完全收敛
- **Modified Policy Iteration**：在评估阶段只做有限次迭代值更新

## 局限性

策略迭代要求已知完整环境模型（转移概率和奖励函数），且仅适用于有限离散状态和动作空间。对于大规模问题，单轮策略评估的计算开销可能很大。

[^src-chapter-4-value-iteration-and-policy-iteration]: [[source-chapter-4-value-iteration-and-policy-iteration]]
[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]: [[source-chapter-3-optimal-state-values-and-bellman-optimality-equation]]
