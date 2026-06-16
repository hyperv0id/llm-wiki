---
title: "Girsanov's Theorem"
type: technique
tags:
  - stochastic-calculus
  - measure-change
  - radon-nikodym
  - path-measure
created: 2026-06-16
last_updated: 2026-06-16
source_count: 1
confidence: high
status: active
---

# Girsanov 定理

Girsanov 定理提供了在连续路径空间上**改变概率测度**的方法，同时保持底层轨迹不变。它是 [[schrodinger-bridge|Schrödinger bridge]] 理论中定义 path measures 之间 KL divergence 的基础[^src-schrodinger-bridges-generative-modeling]。

## 定理陈述

给定 $d$ 维 Brownian motion $(B_t)_{t\in[0,T]}$ 和一个 predictable process $(\theta_s)_{s\in[0,T]}$，定义 density process：

$$Z_t = \exp\left(-\frac{1}{2}\int_0^t \|\theta_s\|^2 ds + \int_0^t \theta_s^\top dB_s\right)$$

则过程：

$$B_t' = B_t - \int_0^t \theta_s ds$$

在新概率测度 $\mathbb{P}'$ 下是 **standard Brownian motion**，且 Radon-Nikodym derivative 为 $\frac{d\mathbb{P}'}{d\mathbb{P}} = Z_T$。

## Novikov 条件

为使 $Z_t$ 成为真正的 martingale（保证测度变换有效）：

$$\mathbb{E}\left[\exp\left(\frac{1}{2}\int_0^T \|\theta_s\|^2 ds\right)\right] < \infty$$

## 从离散时间推导

考虑 Brownian increments $\Delta B_{t_k} \sim \mathcal{N}(0, \Delta t I_d)$ 和 controlled process $X_{t_{k+1}} = X_{t_k} + \sigma u(X_{t_k}, t_k)\Delta t + \sigma\Delta B_{t_k}$。利用 Brownian increment 的 Gaussian 性质得到路径概率比，取 $\Delta t \to 0$：

$$\frac{d\mathbb{P}}{d\sigma\mathbb{B}}(X_{0:T}) = \exp\left(\int_0^T u(X_t,t)dB_t - \frac{1}{2}\int_0^T \|u(X_t,t)\|^2 dt\right)$$

## Path Measure 的 Radon-Nikodym Derivative

对于两个 drift 分别为 $u$ 和 $\tilde{u}$ 的 controlled Itô processes：

$$\frac{d\mathbb{P}^{\tilde{u}}}{d\mathbb{P}^u}(X_{0:T}^u) = \exp\left(-\frac{1}{2}\int_0^T \|(\tilde{u}-u)(X_t^u,t)\|^2 dt + \int_0^T (\tilde{u}-u)(X_t^u,t)^\top dB_t^u\right)$$

## Path-Space KL Divergence

在 $\mathbb{P}^{\tilde{u}}$ 下取期望：

$$\mathrm{KL}(\mathbb{P}^{\tilde{u}}\|\mathbb{P}^u) = \mathbb{E}_{\mathbb{P}^{\tilde{u}}}\left[\frac{1}{2}\int_0^T \|(\tilde{u}-u)(\tilde{X}_t,t)\|^2 dt\right]$$

当 $\mathbb{P}^u = \mathbb{Q}$（零控制）时，退化为：

$$\mathrm{KL}(\mathbb{P}^{\tilde{u}}\|\mathbb{Q}) = \mathbb{E}_{\mathbb{P}^{\tilde{u}}}\left[\frac{1}{2}\int_0^T \|\tilde{u}(\tilde{X}_t,t)\|^2 dt\right]$$

这正是 dynamic SB 所最小化的 **quadratic control cost**。

## 在 SB-SOC 中的应用

最优 bridge measure $\mathbb{P}^\star$ 与 reference $\mathbb{Q}$ 之间的 Radon-Nikodym derivative：

$$\frac{d\mathbb{P}^\star}{d\mathbb{Q}}(X_{0:T}) = e^{-V_T(X_T)+V_0(X_0)} = e^{-\Phi(X_T)+V_0(X_0)}$$

这使得 path-space KL 最小化可以转化为 [[diffusion-schrodinger-bridge-matching]] 和 [[adjoint-matching]] 中使用的 drift-matching objectives。

[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]
