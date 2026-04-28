---
title: "福克-普朗克方程 (Fokker-Planck Equation)"
type: concept
tags:
  - stochastic-processes
  - partial-differential-equations
  - diffusion-models
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# 福克-普朗克方程

**福克-普朗克方程**描述马尔可夫随机过程的概率密度函数 $p(x, t)$ 随时间演化的偏微分方程，是分析扩散模型的理论基石。[^src-tutorial]

## 一般形式

对于非线性朗之万方程：
$$
\dot{\xi} = h(\xi, t) + g(\xi, t) \Gamma(t)
$$

福克-普朗克方程为：

$$
\frac{\partial p(x, t)}{\partial t} = -\frac{\partial}{\partial x} \left[ D^{(1)}(x, t) p(x, t) \right] + \frac{\partial^2}{\partial x^2} \left[ D^{(2)}(x, t) p(x, t) \right]
$$

其中：
- $D^{(1)}(x, t) = h(x, t) + g'(x, t) g(x, t)$（漂移项）
- $D^{(2)}(x, t) = g^2(x, t)$（扩散项）[^src-tutorial]

## 概率流解释

定义概率流 $S(x, t)$：

$$
S(x, t) = D^{(2)}(x, t) \frac{\partial p}{\partial x} - D^{(1)}(x, t) p(x, t)
$$

福克-普朗克方程变为连续性方程：

$$
\frac{\partial p}{\partial t} + \frac{\partial S}{\partial x} = 0
$$

表示概率的守恒。[^src-tutorial]

## 在 SMLD 中的应用

考虑朗之万方程：
$$
\dot{x} = \tau \frac{\partial}{\partial x} \log p(x) + \sigma \Gamma(t)
$$

福克-普朗克分析表明，当 $t \to \infty$ 时，解 $x(t)$ 的分布收敛于 $p(x)$。选择 $\sigma = \sqrt{\tau}$ 可使收敛分布恰好为目标分布。[^src-tutorial]

## 引用

[^src-tutorial]: [[source-chan-2025-diffusion-tutorial]]
