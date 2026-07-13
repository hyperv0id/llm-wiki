---
title: "Label Autocorrelation"
type: concept
tags:
  - time-series-forecasting
  - autocorrelation
  - learning-objective
  - direct-forecast
created: 2026-07-13
last_updated: 2026-07-13
source_count: 2
confidence: high
status: active
---

## Definition

**Label autocorrelation** is the statistical dependence among time steps of a multi-step *label* (future) sequence $Y \in \mathbb{R}^{T \times D}$ given the historical input $L$. Unlike input autocorrelation handled by model architectures (RNN/CNN/Transformer/GNN), label autocorrelation concerns how future values are autoregressively generated: $Y_{t+1}$ depends on $Y_t$ even after conditioning on $L$.[^src-fredf]

## Why It Matters for Direct Forecast

The dominant [[direct-forecast|direct forecast (DF)]] paradigm emits all $T$ steps at once and optimizes step-wise MSE. FreDF formalizes (Theorem 3.1) that this MSE objective equals the conditional negative log-likelihood only under

$$
Y_t \perp Y_{t'} \mid L \quad \forall\, t \neq t'.
$$

Label autocorrelation violates that assumption, so DF training is biased relative to the true multi-step likelihood and forecast quality degrades.[^src-fredf]

## Empirical Evidence

Using double machine learning (DML) with history as confounder (to remove fork-structure pseudo-correlation $Y_t \leftarrow L \rightarrow Y_{t'}$), FreDF measures causal strength $Y_t \to Y_{t'}$ on Weather with $T=192$: ~37.5% of off-diagonal entries exceed 0.3, with periodic banding. After Fourier transform to frequency components $F_k$, only ~3.6% of off-diagonal causations exceed 0.1 — near independence under orthogonal bases.[^src-fredf]

## Mitigations

| Approach | Mechanism | Scope |
|----------|-----------|--------|
| Iterative forecast (IF) | Recursive one-step prediction | Respects label structure but suffers error propagation |
| [[fredf|FreDF]] | Align DF outputs with labels in an orthogonal (e.g., Fourier) domain | Mitigates label dependence under DF; model-agnostic |
| [[qdf|QDF]] | Learn PSD $\Sigma$ in quadratic $L_\Sigma=\|Y-\hat Y\|_{\Sigma^{-1}}^2$ (off-diagonals) | Models residual conditional dependence + [[heterogeneous-task-weights]] |
| [[source-distdf|DistDF]] | Joint-distribution Wasserstein alignment | Avoids likelihood factorization; targets residual [[autocorrelation-bias]] |

FreDF is the first systematic use of frequency analysis to upgrade the *forecast paradigm* (not just architecture) for label autocorrelation.[^src-fredf] Later [[qdf|QDF]] keeps the DF multi-output head but learns a quadratic weighting matrix $\Sigma$ so off-diagonals of $\Sigma^{-1}$ model residual conditional dependence (and diagonals model [[heterogeneous-task-weights|heterogeneous task weights]]).[^src-qdf]

## Related

- [[autocorrelation-bias]] — DistDF's formal bias of MSE under conditional label dependence
- [[frequency-enhanced-direct-forecast]] — FreDF training recipe
- [[source-fredf]], [[source-qdf]]
- [[qdf]], [[quadratic-form-weighted-objective]], [[heterogeneous-task-weights]]

---

[^src-fredf]: [[source-fredf]]
[^src-qdf]: [[source-qdf]]
