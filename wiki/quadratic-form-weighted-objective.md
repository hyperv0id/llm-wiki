---
title: "Quadratic-Form Weighted Objective (QDF)"
type: technique
tags:
  - time-series-forecasting
  - learning-objective
  - quadratic-form
  - meta-learning
  - direct-forecast
  - iclr-2026
created: 2026-07-13
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

## Overview

The **quadratic-form weighted objective** is the training loss used by [[qdf|QDF]] for multi-step [[direct-forecast|direct forecast]] models. Under a Gaussian residual assumption, the conditional NLL of label sequence $Y\in\mathbb{R}^T$ is:

$$
L_\Sigma(X,Y;g_\theta)=\|Y-g_\theta(X)\|_{\Sigma^{-1}}^2=(Y-g_\theta(X))^\top\Sigma^{-1}(Y-g_\theta(X)),
$$

with PSD weighting matrix $\Sigma\in\mathbb{R}^{T\times T}$. Off-diagonal structure of $\Sigma^{-1}$ models [[label-autocorrelation|label autocorrelation]]; non-uniform diagonals implement [[heterogeneous-task-weights|heterogeneous task weights]].[^src-qdf]

## Why Not Fix $\Sigma=I$?

MSE is the special case $\Sigma=I$. True conditional covariance is unknown and hard to estimate from one $Y$ per $X$. Fixed transforms (Fourier in [[fredf|FreDF]], PCA in Time-o1) only ensure *marginal* independence of components and still use equal component weights, so residual conditional dependence and unequal step uncertainty remain.[^src-qdf]

## Learning $\Sigma$ (Bilevel)

QDF treats $\Sigma$ as parameters optimized for *generalization*, not as a sample covariance estimate:

$$
\min_{\Sigma\succeq 0}\, L_\Sigma(X^{\mathrm{out}},Y^{\mathrm{out}};g_{\theta^\star})\quad
\text{s.t.}\quad
\theta^\star=\arg\min_\theta L_\Sigma(X^{\mathrm{in}},Y^{\mathrm{in}};g_\theta).
$$

- **Reparameterization:** $\Sigma=LL^\top$ with softplus-positive diagonals of lower-triangular $L$ → unconstrained optimization.
- **Atomic update (Alg. 1):** split $D\to D_{\mathrm{in}},D_{\mathrm{out}}$; $N$ inner GD steps on $\theta$; one outer step $\Sigma\leftarrow\Sigma-\nabla_\Sigma L_\Sigma(\cdot;\theta)$ with gradient *through* the updated $\theta$.
- **Workflow (Alg. 2):** init $\Sigma=I$; chronological $K$-way train splits; iterate Alg. 1 over folds until $\|\Sigma_{n+1}-\Sigma_n\|_F<10^{-4}$ or $N_{\mathrm{out}}$; finally minimize $L_\Sigma$ on full $D_{\mathrm{train}}$.[^src-qdf]

No test leakage: only training data is used. Extra cost is training-only; inference is standard DF forward pass.[^src-qdf]

## Ablation Semantics

| Variant | Diagonals | Off-diagonals | Role |
|---------|-----------|---------------|------|
| DF (MSE) | fixed 1 | fixed 0 | baseline |
| QDF† | learned | fixed 0 | hetero task weights only |
| QDF‡ | fixed 1 | learned | autocorrelation only |
| QDF | learned | learned | both (best) |

Empirical synergy: both factors improve over DF; full matrix wins on ETT/ECL/Weather.[^src-qdf]

## Relation to Meta-Learning

Treating $\Sigma$ as learnable is related to MAML/Reptile-style bilevel optimization, but goals differ: QDF seeks a *static* forecasting objective for one task (with holdout from the same task), not fast adaptation across many tasks. MAML/iMAML/MAML++/Reptile can still optimize $\Sigma$ and beat DF, yet underperform QDF’s generalization-targeted outer loop on ECL.[^src-qdf]

## Related

- Entity: [[qdf]]
- Source: [[source-qdf]]
- Contrast: [[frequency-enhanced-direct-forecast]] (fixed Fourier mix), [[joint-distribution-wasserstein-alignment]] (OT, non-likelihood)
- Concepts: [[label-autocorrelation]], [[heterogeneous-task-weights]], [[autocorrelation-bias]]

---

[^src-qdf]: [[source-qdf]]
