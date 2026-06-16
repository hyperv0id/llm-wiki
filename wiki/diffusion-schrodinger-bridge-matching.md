---
title: "Diffusion Schrödinger Bridge Matching"
type: technique
tags:
  - schrödinger-bridge
  - generative-modeling
  - iterative-markovian-fitting
  - diffusion-models
  - dsbm
created: 2026-06-16
last_updated: 2026-06-16
source_count: 1
confidence: medium
status: active
---

# Diffusion Schrödinger Bridge Matching

Diffusion Schrödinger Bridge Matching (DSBM, Shi et al. 2023) 将 [[iterative-markovian-fitting]] (IMF) 参数化实现：通过参数化 forward 和 reverse Markov drift 求解动态 [[schrodinger-bridge]] 问题[^src-schrodinger-bridges-generative-modeling]。它将 denoising diffusion 和 flow matching 推广到任意边际分布，无需固定前向噪声过程。

## 前向 Markovian 投影 SDE

前向 Markovian 投影构造一个 SDE，其在时刻 $t$ 的 marginal 与 bridge mixture $\Pi^{2n}$ 匹配，生成 Markov process $\mathbb{M}^{2n+1}$：

$$dX_t = [f(X_t,t) + \sigma_t^2 \mathbb{E}_{\Pi_{T|t}}[\nabla \log Q_{T|t}(X_T|X_t)|X_t]]dt + \sigma_t dB_t$$

This SDE converges to the optimal SB forward dynamics driven by $u^\star = \sigma_t \nabla \log \varphi_t$, but uses conditional score matching on the reference transition density $Q_{T|t}$ rather than requiring access to the unknown Schrödinger potential $\varphi_t$.

## Parameterization

The control drift is parameterized by $u_\theta(x,t) \approx \sigma_t \mathbb{E}_{\Pi_{T|t}}[\nabla \log Q_{T|t}(X_T|X_t)|X_t=x]$. Similarly, the reverse drift $u_\phi$ approximates $\sigma_t \mathbb{E}_{\Pi_{0|t}}[\nabla \log Q_{t|0}(X_t|X_0)|X_t=x]$.

## DSBM Losses

### Forward DSBM Loss

$$\mathcal{L}_{\text{DSBM}}(\theta) = \int_0^T \mathbb{E}_{\Pi_{t,T}}\left[\|\sigma_t \nabla \log Q_{T|t}(X_T|X_t) - u_\theta(X_t,t)\|^2\right]dt$$

This corresponds to the Markovian projection $\mathrm{proj}_\mathcal{M}$: given samples from a bridge mixture $\Pi$, learn the drift of the closest Markov process.

### Reverse DSBM Loss

$$\mathcal{L}_{\text{DSBM}}(\phi) = \int_0^T \mathbb{E}_{\Pi_{t,0}}\left[\|\sigma_t \nabla \log Q_{t|0}(X_t|X_0) - u_\phi(X_t,t)\|^2\right]dt$$

Symmetric to the forward loss, but matches the reverse-time conditional scores.

### Consistency Loss

Enforces time-reversal symmetry between forward and reverse SDEs: if $u_\theta$ drives $X_0 \to X_T$ and $u_\phi$ drives the time-reversed process, they must satisfy $u_\theta + u_\phi = \sigma_t \nabla \log \Pi_t^{2n}$ (the score of the current bridge mixture). The consistency loss is:

$$\mathcal{L}_{\text{cons}}(\theta,\phi) = \int_0^T \mathbb{E}_{\Pi_t^{2n}}\left[\|u_\theta(X_t,t) + u_\phi(X_t,t) - \sigma_t \nabla \log \Pi_t^{2n}(X_t)\|^2\right]dt$$

where the bridge mixture score decomposes as:

$$\nabla \log \Pi_t^{2n}(x) = \mathbb{E}_{\Pi_{T|t}}[\nabla \log Q_{T|t}(X_T|X_t)|X_t=x] + \mathbb{E}_{\Pi_{0|t}}[\nabla \log Q_{t|0}(X_t|X_0)|X_t=x]$$

### Total Joint Loss

$$\mathcal{L}_{\text{DSBM}}(\theta,\phi) = \mathcal{L}_{\text{DSBM}}(\theta) + \mathcal{L}_{\text{DSBM}}(\phi) + \lambda \mathcal{L}_{\text{cons}}(\theta,\phi)$$

The hyperparameter $\lambda$ trades off individual projection fidelity against mutual consistency.

## Convergence

At convergence, $u_\theta^\star = u^\star = \sigma_t \nabla \log \varphi_t$ recovers the optimal SB control drift. The Pythagorean identity in path space guarantees monotonic KL decrease:

$$\mathrm{KL}(\mathbb{P}^n\|\mathbb{P}^\star) \geq \mathrm{KL}(\mathbb{P}^{n+1}\|\mathbb{P}^\star), \quad \lim_{n\to\infty} \mathrm{KL}(\mathbb{P}^n\|\mathbb{P}^\star) = 0$$

## Relation to [[sinkhorn-algorithm|Sinkhorn's Algorithm]]

DSBM is the dynamic counterpart of the static Sinkhorn algorithm. While Sinkhorn alternates KL projections between marginal constraint sets in the space of couplings $\Pi(\pi_0,\pi_T)$, DSBM alternates Markovian and reciprocal projections in the space of path measures $\mathcal{P}(C([0,T];\mathbb{R}^d))$. Both converge to their respective SB solutions via alternating projections.

## Comparison with Score Matching

| Aspect | DSBM | Standard Score Matching |
|--------|------|------------------------|
| Prior | Arbitrary $\pi_0$ | Usually Gaussian |
| Target | Arbitrary $\pi_T$ | Data distribution |
| Forward process | Learned, controlled by $u_\theta$ | Fixed (VE/VP/OU) |
| Training | Iterative IMF (multiple rounds) | One-pass score estimation |
| Sampling | SDE with learned $u_\theta$ | Reverse SDE with $s_\theta$ |

[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]
