---
title: "Adjoint Schrödinger Bridge Sampler"
type: technique
tags:
  - schrödinger-bridge
  - sampling
  - boltzmann-distribution
  - adjoint-matching
created: 2026-06-16
last_updated: 2026-06-16
source_count: 1
confidence: medium
status: active
confidence: high
status: active
---

# Adjoint Schrödinger Bridge Sampler

Liu et al. (2025) 的方法，用于从未归一化的 Boltzmann distribution $\pi_T(x) = e^{-U(x)}/Z$ 采样[^src-schrodinger-bridges-generative-modeling]。

交替优化 forward half-bridge（[[adjoint-matching|SB-AM]]）和 backward half-bridge（corrector matching）。

## SB-AM Loss for Sampling

$$\mathcal{L}_{\mathrm{SB-AM}}(u) = \mathbb{E}_{\mathbb{P}^{\bar{u}}}\!\left[\frac{1}{2}\int_0^T \|u + \sigma_t(\nabla U + \nabla\log\hat{\varphi}_T)\|^2 dt\right]$$

## Corrector Matching

$$\mathcal{L}_{\mathrm{SB-CM}}(\hat{Z}_T) = \mathbb{E}_{p^{\bar{u}}_{0,T}}[\|\hat{Z}_T(X_T) - \nabla\log Q_{T|0}(X_T|X_0)\|^2]$$

## Forward Half-Bridge 定理

优化 $\mathcal{L}_{\mathrm{SB-AM}}$ 求解 $\mathbb{P}^u = \arg\min\{\mathrm{KL}(\mathbb{P}^u\|\mathbb{P}^{\hat{Z}}) : p_0=\pi_0\}$。Backward half-bridge 定理的对应公式。

## 关键优势

不需要目标分布 $\pi_T$ 的显式样本，仅需 energy evaluation $\nabla U(x)$。使用 replay buffer 存储 $(X_T, \nabla U, \nabla\log\hat{\varphi}_T)$。交替优化结构类似 [[sinkhorn-algorithm]]。

## 参见

- [[adjoint-matching]]
- [[sinkhorn-algorithm]]
- [[schrodinger-bridge]]

[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]
