---
title: "Adjoint Matching"
type: technique
tags:
  - schrödinger-bridge
  - stochastic-optimal-control
  - adjoint-methods
  - generative-modeling
created: 2026-06-16
last_updated: 2026-06-16
source_count: 1
confidence: medium
status: active
confidence: high
status: active
---

# Adjoint Matching

Adjoint Matching (Domingo-Enrich et al. 2024) 通过 matching objectives 学习最优 [[schrodinger-bridge]] 控制，避免 full path-space KL divergence[^src-schrodinger-bridges-generative-modeling]。它将 [[stochastic-optimal-control-sb|SB 表述为 SOC]]，利用 adjoint sensitivity method 导出基于梯度的损失。

## Lean Adjoint Matching Objective

主损失将控制 $u$ 与由 terminal cost gradient 构造的目标匹配，在 stopped-gradient process $\bar{u} = \text{stopgrad}(u)$ 下求值：

$$\mathcal{L}_{\text{simple-AM}}(u) = \mathbb{E}_{X_{0:T}\sim\mathbb{P}^{\bar{u}}}\left[\frac{1}{2}\int_0^T \|u(X_t,t) + \sigma_t \nabla_{x_T}\Phi(X_T)\|^2 dt\right]$$

The functional derivative of this loss with respect to the control $u$ is:

$$\frac{\delta}{\delta u}\mathcal{L}_{\text{AM}}(u)(x,t) = u(x,t) + \sigma_t \mathbb{E}_{\mathbb{P}^{\bar{u}}}[a(X_{t:T},t)|X_t=x]$$

where $a$ is the **adjoint state** satisfying the backward SDE:

$$da_t = -\nabla(f + \sigma_t\bar{u})^\top a_t dt + \sigma_t \nabla c(X_t,t)dt, \quad a_T = \nabla \Phi(X_T)$$

Here $f$ is the reference drift, $c$ is the running cost, and $\Phi$ is the terminal cost. The adjoint state $a_t$ propagates gradient information backward through the stochastic dynamics.

## Critical Point

At a critical point of the AM loss, the learned control recovers the optimal SB control:

$$u^\star(x,t) = -\sigma_t \mathbb{E}_{\mathbb{P}^\star}[\nabla \Phi(X_T)|X_t=x]$$

This is the conditional expectation of the terminal cost gradient under the optimal path measure.

## SB Adjoint Matching (SB-AM)

For the specific Schrödinger bridge setting, the terminal cost $\Phi$ is derived from the Schrödinger potentials:

$$\Phi(x) = \log\frac{\hat{\varphi}_T(x)}{\pi_T(x)} = \log \hat{\varphi}_T(x) - \log \pi_T(x)$$

$$\nabla\Phi(x) = \nabla\log\hat{\varphi}_T(x) - \nabla\log\pi_T(x)$$

This decomposes the terminal cost gradient into the unknown Schrödinger potential gradient $\nabla\log\hat{\varphi}_T$ and the known target score $\nabla\log\pi_T$.

## Corrector Matching

Since $\nabla\log\hat{\varphi}_T$ is unknown, it must be learned. Corrector matching trains a network $\hat{Z}_T$ to approximate it:

$$\mathcal{L}_{\text{SB-CM}}(\hat{Z}_T) = \mathbb{E}_{p^\star_{0,T}}\left[\|\hat{Z}_T(X_T) - \nabla_{x_T}\log Q_{T|0}(X_T|X_0)\|^2\right]$$

The minimizer satisfies $\hat{Z}_T^\star = \nabla\log\hat{\varphi}_T$, providing the missing piece for the SB-AM terminal cost.

## Adjoint Sampling Algorithm

The training procedure alternates between data collection and optimization:

1. **Collection phase**: Simulate the forward SDE with current control $\bar{u}$ to collect terminal pairs $(X_T, \nabla\Phi(X_T))$ in a replay buffer.

2. **Training phase**: Sample $X_t \sim Q_{t|T}(\cdot|X_T)$ (reference bridge conditioning) and optimize $\mathcal{L}_{\text{RAM}}$ (replay-based adjoint matching).

3. **Fixed point**: Iterate until convergence. The fixed point

   $$u = \text{proj}(u) = u - \frac{\delta}{\delta u}\mathcal{L}_{\text{AM}}(u)$$

   satisfies $u = u^\star$, recovering the exact SB control.

## Comparison with DSBM

| Aspect | Adjoint Matching | [[diffusion-schrodinger-bridge-matching|DSBM]] |
|--------|-----------------|------|
| Core mechanism | Adjoint state + terminal cost matching | Markovian/reciprocal projection alternation |
| Gradient computation | Backward SDE for adjoint $a_t$ | Direct score matching on $Q_{T|t}, Q_{t|0}$ |
| Convergence | Fixed-point iteration of $\text{proj}(u)$ | Alternating projection Pythagorean descent |
| Terminal cost | $\Phi = \log(\hat{\varphi}_T/\pi_T)$ | Not explicitly parameterized |
| Auxiliary learner | $\hat{Z}_T$ for corrector matching | None required |

## Connection to [[adjoint-schrodinger-bridge-sampler]]

The Adjoint Schrödinger Bridge Sampler builds on adjoint matching to construct efficient sampling procedures that avoid full trajectory simulation, using the learned control $u^\star$ to directly generate samples from $\pi_T$ starting from $\pi_0$.

[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]