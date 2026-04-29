---
title: "压缩映射定理"
type: concept
tags:
  - reinforcement-learning
  - fixed-point
  - contraction-mapping
created: 2026-04-27
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# 压缩映射定理

压缩映射定理（Contraction Mapping Theorem），又称巴拿赫不动点定理（Banach Fixed-Point Theorem），是函数分析和强化学习理论中的重要数学工具。[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]

## 形式化定义

设 $(X, \|\cdot\|)$ 是完备度量空间，算子 $T: X \to X$ 满足：存在 $\beta \in [0,1)$ 使得对任意 $x, y \in X$：

$$\|T(x) - T(y)\| \leq \beta \|x - y\|$$

则 $T$ 称为**压缩映射**（contraction）。定理结论：
1. $T$ 存在唯一不动点 $x^* = T(x^*)$
2. 从任意初始点 $x_0 \in X$ 出发，迭代 $x_{k+1} = T(x_k)$ 收敛到 $x^*$
3. 收敛速度为几何级数：$\|x_k - x^*\| \leq \beta^k \|x_0 - x^*\|$

## 在强化学习中的应用

### 最优性算子的压缩性

第 3 章证明了贝尔曼最优性算子 $T^*$（在无穷范数 $\|\cdot\|_\infty$ 下）是压缩的：

$$\|T^*(V_1) - T^*(V_2)\|_\infty \leq \gamma \|V_1 - V_2\|_\infty$$

其中 $T^*(V)(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s') \right]$ 且 $\gamma \in [0,1)$ 为折扣因子。

由此直接得到：[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]
- [[bellman-optimality-equation|贝尔曼最优方程]] 存在唯一解 $v_*$
- 从任意初始值出发，迭代都收敛到 $v_*$

### 对值迭代的影响

[[value-iteration|值迭代]] 正是压缩映射迭代的直接体现。收敛速度为 $\gamma^k \|V_0 - v_*\|_\infty$，即收敛率取决于折扣因子 $\gamma$。$\gamma$ 越小收敛越快，但求解中过于短视。

### 对策略评估的影响

贝尔曼算子 $T_\pi(V) = r_\pi + \gamma P_\pi V$ 在折扣因子 $\gamma \in [0,1)$ 条件下也是压缩的。这是策略评估使用迭代法求解的理论保证。

## 意义

压缩映射定理将强化学习中的值函数求解统一为**不动点迭代**框架，简洁地解释了为什么值迭代、策略评估等算法都能可靠收敛。[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]

[^src-chapter-3-optimal-state-values-and-bellman-optimality-equation]: [[source-chapter-3-optimal-state-values-and-bellman-optimality-equation]]
