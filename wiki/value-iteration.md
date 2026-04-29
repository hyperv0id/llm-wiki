---
title: "值迭代"
type: technique
tags:
  - reinforcement-learning
  - dynamic-programming
  - value-iteration
created: 2026-04-27
last_updated: 2026-04-28
source_count: 2
confidence: high
status: active
---

# 值迭代

值迭代（Value Iteration）是直接求解 [[bellman-optimality-equation|贝尔曼最优方程]] 的动态规划算法，用于在已知环境模型的情况下找到最优策略。[^src-chapter-4-value-iteration-and-policy-iteration]

## 算法描述

值迭代的核心思想是迭代应用最优性算子：

$$V_{k+1}(s) = \max_a \left[ \sum_{s'} p(s'|s,a) \left( r(s,a,s') + \gamma V_k(s') \right) \right]$$

算法流程：
1. 初始化值函数 $V_0$（任意值）
2. 对每个状态 $s$，根据当前值函数计算最优动作的期望值
3. 重复步骤 2 直到收敛

每轮迭代先执行贪心改进，再更新值函数；反复执行后，值函数收敛到最优值函数 $v_*$。[^src-chapter-4-value-iteration-and-policy-iteration]

## 收敛性保证

值迭代建立在最优性算子的**压缩映射**性质之上。根据 [[contraction-mapping-theorem|压缩映射定理]]，当折扣因子 $\gamma \in [0,1)$ 时，最优性算子在任何范数下都是压缩的，因此：

- 从任意初始值 $V_0$ 出发，���代都收敛到唯一不动点
- 收敛速度为几何级数：$\|V_k - v_*\| \leq \gamma^k \|V_0 - v_*\|$

这保证了值迭代的可靠收敛性。[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]

## 提取策略

值迭代收敛后，可通过贪心策略提取最优策略：

$$\pi_*(s) = \arg\max_a \sum_{s'} p(s'|s,a) \left( r(s,a,s') + \gamma v_*(s') \right)$$

## 与策略迭代的比较

| 特性 | 值迭代 | [[policy-iteration|策略迭代]] |
|------|--------|----------------------|
| 单轮成本 | 较低（一次扫描） | 较高（需完整策略评估） |
| 迭代轮数 | 较多 | 较少 |
| 每次迭代更新 | 所有状态 | 仅评估后的策略 |

值迭代与策略迭代是 [[truncated-policy-iteration|截断策略迭代]] 的两个端点：评估步数为 1 时退化为值迭代，评估步数足够大时逼近策略迭代。[^src-chapter-4-value-iteration-and-policy-iteration]

## 变体

- **异步值迭代**：不要求每次扫描所有状态，可按任意顺序更新
- ** Gauss-Seidel 迭代**：使用新计算出的值立即更新相邻状态
- **优先级扫描**：优先更新值变化最大的状态

[^src-chapter-4-value-iteration-and-policy-iteration]: [[source-chapter-4-value-iteration-and-policy-iteration]]
[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]: [[source-chapter-3-optimal-state-values-and-bellman-optimality-equation]]
