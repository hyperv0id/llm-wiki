---
title: "TimeDiT Physics-Informed Sampling"
type: technique
tags:
  - diffusion
  - physics-informed
  - pde
  - energy-based-model
  - langevin-dynamics
  - time-series
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

**Physics-Informed TimeDiT** is a finetuning-free model editing strategy that incorporates physical knowledge — represented as partial differential equations (PDEs) — into the diffusion sampling process of [[timedit|TimeDiT]][^src-timedit]. By guiding the reverse diffusion with gradients from physical laws, generated samples satisfy known PDEs without requiring model retraining or parameter updates.

## Motivation

Real-world time series are often governed by underlying physical principles (e.g., fluid dynamics, heat transfer, wave propagation). Incorporating this knowledge enhances model performance and interpretability, especially in data-scarce domains[^src-timedit]. Traditional approaches require training dedicated models for each physical system — time-consuming and impractical when multiple physical laws interact.

## Mechanism

### Energy-Based Optimization

Given a pre-trained diffusion model `p(x_tar | x_con)` and a physics residual function `K(x_tar; F)` measuring PDE consistency, the refined distribution `q(x_tar | x_con)` is obtained by[^src-timedit]:

```
q = argmax E_{x∼q}[K(x; F)] - α·D_KL(q || p)
```

The first term maximizes physical consistency (residual → 0), the second controls divergence from the learned distribution (α balances the trade-off).

### Closed-Form Solution (Theorem 3.1)

The optimization has a closed-form solution: **Boltzmann distribution** on energy function `E(x_tar; x_con) = K(x_tar; F) + α·log p(x_tar | x_con)`[^src-timedit]:

```
q(x_tar | x_con) = (1/Z) · exp(K(x_tar; F) + α·log p(x_tar | x_con))
```

where Z is the partition function. This means incorporating physics knowledge is equivalent to sampling from this Boltzmann distribution — no parameter updates needed[^src-timedit].

### Langevin Dynamics Sampling

The Boltzmann distribution is sampled via Langevin dynamics after standard DDPM reverse diffusion (Algorithm 1)[^src-timedit]:

```
x_{j+1} = x_j + ε·∇K(x_j; x_con) + α·ε·∇log p(x_j | x_con) + √(2ε)·σ
```

where:
- `ε·∇K`: Physical gradient pushing toward PDE satisfaction
- `α·ε·∇log p`: Distribution gradient keeping samples realistic
- `√(2ε)·σ`: Stochastic noise for exploration

The log-likelihood is approximated as `log p(x_tar | x_con) ≈ -E_{ε,t}[||ε_θ(x_tar, t; x_con) - ε||²]` — the denoising score matching objective[^src-timedit].

## Physics Residual Function

For a PDE describing signal evolution over spatial coordinate `u`[^src-timedit]:

```
∂x/∂t = F(t, x, u, ∂x/∂u_i, ∂²x/(∂u_i ∂u_j), ...)
```

The consistency metric is the squared residual:

```
K(x_tar; F) = -||∂x_tar/∂t - F(...)||²
```

K reaches its maximum (0) when the predicted series perfectly satisfies the PDE[^src-timedit].

## Experimental Validation

Tested on four PDE systems (Burgers, Advection, Diffusion-Reaction, Kolmogorov Flow). **Zero-shot** TimeDiT with physics-informed sampling outperforms **fully trained** baselines (DLinear, PatchTST, NeuralCDE) that were trained on 5,000 PDE-generated samples each[^src-timedit]:

| PDE System | TimeDiT (Zero-shot) | DLinear (Full-shot) | PatchTST (Full-shot) |
|------------|---------------------|---------------------|---------------------|
| Burgers MSE | **0.011** | 0.031 | 0.029 |
| Vorticity MSE | **1.524** | 2.650 | 2.651 |

A single TimeDiT checkpoint with physics-informed sampling replaces 18 task-specific models (3 baselines × 6 PDE equations), with superior accuracy and dramatically lower computational overhead[^src-timedit].

## Key Properties

1. **Finetuning-free**: No gradient-based parameter updates — physics integration happens purely during inference through Langevin dynamics[^src-timedit].
2. **Flexible**: Different physical constraints (different PDEs) can be applied to the same pre-trained model by changing the K function — no model retraining[^src-timedit].
3. **Theoretically grounded**: Closed-form Boltzmann solution provides formal guarantees on the optimality of the physics-refined distribution[^src-timedit].
4. **Trade-off control**: α parameter controls balance between physical consistency and distribution fidelity[^src-timedit].

## Limitations

- Requires known PDE form; cannot infer physical laws from data
- Langevin dynamics add k extra sampling steps beyond standard denoising
- Only tested on governed PDE systems; applicability to implicit/unknown physical constraints unclear

## Related Pages

- [[timedit]] — TimeDiT model overview
- [[timedit-masking]] — Unified masking mechanism
- [[diffusion-models]] — Diffusion model fundamentals
- [[energy-based-model]] — Energy-based models
- [[langevin-dynamics]] — Langevin dynamics for MCMC sampling
- [[dyffusion]] — DYffusion, another physics-informed diffusion approach
- [[elbo]] — ELBO, log-likelihood approximation foundation

[^src-timedit]: [[source-timedit]]
