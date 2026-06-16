---
title: "Fractional Schrödinger Bridge"
type: concept
tags:
  - schrödinger-bridge
  - fractional-brownian-motion
  - doob-h-transform
  - long-range-dependence
  - markov-approximation
created: 2026-06-16
last_updated: 2026-06-16
source_count: 1
confidence: medium
status: active
---

# Fractional Schrödinger Bridge

The fractional Schrödinger bridge replaces the standard Brownian motion reference process with **fractional Brownian motion** (fBM), capturing long-range temporal dependencies that Markovian processes cannot model[^src-schrodinger-bridges-generative-modeling]. The Hurst index $H \in (0,1)$ controls the strength and sign of temporal correlations.

## Fractional Brownian Motion

fBM $B_t^H$ is a continuous, zero-mean Gaussian process with covariance $\mathbb{E}[B_t^H B_s^H] = \frac{1}{2}(|t|^{2H} + |s|^{2H} - |t-s|^{2H})$. Its increments are:
- **Positively correlated** for $H > 1/2$ (persistence: trends persist)
- **Independent** for $H = 1/2$ (recovers standard Brownian motion)
- **Negatively correlated** for $H < 1/2$ (anti-persistence: mean reversion)

Crucially, fBM is **non-Markov** for $H \neq 1/2$: the future depends on the entire past history, not just the current state. This breaks the standard SB machinery that relies on Markovian SDEs.

## Mandelbrot-Van Ness Representation

fBM can be expressed as a weighted integral of standard Brownian motion:

$$B_t^H = \frac{1}{\Gamma(H+1/2)}\int_0^t (t-s)^{H-1/2} dB_s$$

The kernel $(t-s)^{H-1/2}$ encodes the long-range memory: the weight decays algebraically rather than exponentially, creating persistent influence from the distant past.

## Markov Approximation via OU Processes

To make fBM computationally tractable within the SB framework, approximate it by a superposition of $K$ Ornstein-Uhlenbeck (OU) processes:

$$\hat{B}_t^H = \sum_{k=1}^K \omega_k Y_t^k, \quad dY_t^k = -\gamma_k Y_t^k dt + dB_t$$

with decay rates $\gamma_k = r k^{-n}$ for $r>1$ and $n = \frac{K+1}{2}$. Each OU process $Y_t^k$ captures a different temporal scale; together they approximate the algebraic memory kernel with a sum of exponentials.

### Augmented State

The augmented state combines the original process with all OU components:

$$Z_t = (\hat{X}_t, Y_t^1, \ldots, Y_t^K) \in \mathbb{R}^{d \times (K+1)}$$

This joint state follows a **linear Markov SDE**:

$$dZ_t = F Z_t dt + \sigma_t dB_t$$

where the matrix $F \in \mathbb{R}^{(K+1) \times (K+1)}$ encodes the coupling between the original state $\hat{X}_t$ and each $Y_t^k$:
- The $(1,1)$ block governs $\hat{X}_t$'s own dynamics.
- The $(1, k+1)$ entries couple $\hat{X}_t$ to $Y_t^k$ via weights $\omega_k$.
- The $(k+1, k+1)$ entries are $-\gamma_k$ (mean reversion of each OU process).

The augmented system is Markovian, so standard SB techniques apply in the $d(K+1)$-dimensional augmented space.

## Fractional Brownian Bridge via Doob's h-Transform

To condition the augmented process on hitting a target $x_T$ at time $T$, apply [[doob-h-transform|Doob's h-transform]] with

$$h(z,t) = S_{T|t}(x_T|Z_t = z)$$

yielding the conditioned SDE:

$$dZ_{t|0,T} = \left(F Z_{t|0,T} + \sigma_t^2 \nabla_z \log S_{T|t}(x_T|Z_{t|0,T}=z)\right)dt + \sigma_t dB_t$$

where $S_{T|t}$ is the transition density of the augmented (linear, Gaussian) system.

### Closed-Form Gradient

Because the augmented system is linear-Gaussian, $S_{T|t}$ has a closed form and its gradient is analytically tractable. For each coordinate $i$ of $\hat{X}_T$:

$$\nabla_z \log S_{T|t}^i(x_T|z) = [1, \omega_1 \zeta_1(t,T), \ldots, \omega_K \zeta_K(t,T)]^\top \frac{x_T^i - \mu_{T|t}^i(z)}{\sigma_{T|t}^2}$$

where:

$$\begin{aligned}
\zeta_k(t,T) &= e^{-\gamma_k(T-t)} - 1 \\
\mu_{T|t}(z) &= x + \sum_{k=1}^K \omega_k y_k \zeta_k(t,T) \\
\sigma_{T|t}^2 &= \varepsilon \sum_{k,\ell} \frac{\omega_k \omega_\ell}{\gamma_k+\gamma_\ell}\left[1-e^{-(\gamma_k+\gamma_\ell)(T-t)}\right]
\end{aligned}$$

The vector $[1, \omega_1\zeta_1, \ldots, \omega_K\zeta_K]$ gives the sensitivity of $\hat{X}_T$ to each component of the augmented state. The term $\frac{x_T^i - \mu_{T|t}^i}{\sigma_{T|t}^2}$ is the standardized residual: how far the target is from the expected position, scaled by the conditional variance.

## Properties and Applications

| Property | Standard SB ($H=1/2$) | Fractional SB ($H \neq 1/2$) |
|----------|----------------------|------------------------------|
| Reference | Markovian (Brownian) | Non-Markov (fBM) |
| Memory | None | Long-range (algebraic decay) |
| Dimensionality | $d$ | $d(K+1)$ (augmented) |
| Bridge construction | Standard Doob h-transform | Augmented Doob h-transform |
| Gradient formula | $x_T - x$ (linear) | Weighted sum over $K$ OU scales |

Fractional SB is particularly relevant for applications where temporal dependencies span long horizons: climate modeling, financial time series, and physical systems with memory effects. The OU approximation trades exactness for computational tractability, with larger $K$ providing better approximations of true fBM behavior.

## Connection to Building Methods

The fractional SB construction uses [[building-schrodinger-bridges|Doob's h-transform]] in the augmented state space. The [[schrodinger-bridge|Schrödinger potentials]] now operate on $(z,t)$ rather than $(x,t)$, but the underlying Hopf-Cole structure remains intact. Sampling requires simulating the $d(K+1)$-dimensional augmented SDE, then projecting back to the $d$-dimensional physical state.

[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]