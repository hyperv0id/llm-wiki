---
title: "Doob's h-Transform"
type: technique
tags:
  - schrödinger-bridge
  - stochastic-processes
  - probability-tilting
  - doob-transform
created: 2026-06-16
last_updated: 2026-06-16
source_count: 1
confidence: high
status: active
---

# Doob's h-Transform

核心思想：对 reference process 做概率倾斜（tilting）使满足端点条件[^src-schrodinger-bridges-generative-modeling]。

给定 $h(x,t) = \mathbb{E}[h(X_{t+\Delta t},t+\Delta t)|X_t=x]$ 满足 martingale property，变换后 SDE：

$$dX_t^h = (f + \sigma_t^2\nabla\log h(X_t^h,t))dt + \sigma_t dB_t$$

Reweighted path measure：

$$\mathbb{P}^h(X_{0:T}) = \frac{h(X_T,T)}{h(X_0,0)}\mathbb{Q}(X_{0:T})$$

## 与 SB 的关系

设 $h(x,t) = \mathbb{E}_Q[\varphi_T(X_T)|X_t=x]$，则 $u^\star = \sigma_t\nabla\log h = \sigma_t\nabla\log\varphi_t$，恢复 [[schrodinger-bridge|SB]] 最优控制。

## 在 Fractional Schrödinger Bridge 中

在 [[fractional-schrodinger-bridge]] 中：$h(z,t) = S_{T|t}(x_T|z)$ 其中 $z$ 包含 OU auxiliary processes，产生 drift correction $\sigma_t^2\nabla_z\log S_{T|t}(x_T|z)$。

## 参见

- [[schrodinger-bridge]]
- [[fractional-schrodinger-bridge]]
- [[iterative-markovian-fitting]]

[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]
