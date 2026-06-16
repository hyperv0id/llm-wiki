---
title: "Conditional Score and Flow Matching"
type: technique
tags:
  - schrödinger-bridge
  - flow-matching
  - score-matching
  - entropic-optimal-transport
created: 2026-06-16
last_updated: 2026-06-16
source_count: 1
confidence: medium
status: active
confidence: high
status: active
---

# Conditional Score and Flow Matching

[SF]²M（Tong et al. 2023）提供了一种无需仿真的 SB 学习方法，仅需 $\pi_0, \pi_T$ 的 empirical samples[^src-schrodinger-bridges-generative-modeling]。

## 核心思想

先求解 entropic OT 获得最优 coupling $\pi_{0,T}^\star$，则 dynamic SB 就是 Brownian bridge 的混合（Mac 分解）。这意味着学习 SB 可以分解为两步：先解 static EOT 得到端点配对，再学习 conditional bridges。

## Conditional Control Drift 和 Score

端点 $(x_0, x_T)$ 之间的 Brownian bridge：

$$u(x,t|x_0,x_T) = \frac{1-2t}{t(1-t)}(x - (tx_1+(1-t)x_0)) + (x_1-x_0)$$

$$\nabla\log p_t(x|x_0,x_T) = \frac{tx_1 + (1-t)x_0 - x}{\sigma_t^2 t(1-t)}$$

前者是 conditional control drift，后者是 conditional score function。两者共同刻画了给定端点条件下 bridge 的局部行为。

## Conditional Objective

Conditional objective 与 unconditional objective 具有相同的梯度：

$$L_{\mathrm{[SF]^2M}}(\theta) = \int_0^T \mathbb{E}\left[\|v_\theta - u(\cdot|x_0,x_T)\|^2 + \lambda^2\|s_\theta - \nabla\log p_t(\cdot|x_0,x_T)\|^2\right]dt$$

其中期望在 $(x_0,x_T) \sim \pi_{0,T}^\star$ 和 $x \sim p_t(\cdot|x_0,x_T)$ 上取。这个 formulation 的关键优势在于：conditional targets 有闭式表达，无需仿真即可直接计算。

## 退化与恢复

当 coupling 选用 product measure $\pi_0 \otimes \pi_T$ 时，[SF]²M 退化为 classical flow matching。当 $\pi_{0,T}^\star$ 为 entropic OT coupling 时，恢复完整的 SB。这一性质揭示了 flow matching 和 SB 之间的连续谱：通过调节 coupling 的 entropy regularization 程度，可以在 deterministic OT 和 independent coupling 之间平滑插值。

## 参见

- [[schrodinger-bridge]] — SB 的一般理论
- [[building-schrodinger-bridges]] — SB 的构造方法综述

[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]