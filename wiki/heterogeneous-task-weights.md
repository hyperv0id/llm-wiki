---
title: "Heterogeneous Task Weights (Multi-Step Forecasting)"
type: concept
tags:
  - time-series-forecasting
  - learning-objective
  - multi-task-learning
  - direct-forecast
  - iclr-2026
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: medium
status: active
---

## Definition

**Heterogeneous task weights** refer to assigning *non-uniform* importance to different future-horizon steps when training a multi-step [[direct-forecast|direct forecast]] model. Predicting each step $t=1,\ldots,T$ is treated as a distinct multitask head; steps differ in difficulty and residual uncertainty, so equal weighting (as in MSE) is suboptimal.[^src-qdf]

## Formalization in QDF

Under the Gaussian NLL view, the quadratic objective $L_\Sigma=\|Y-\hat Y\|_{\Sigma^{-1}}^2$ uses the **diagonal** of $\Sigma^{-1}$ as per-step weights. Non-uniform diagonals encode heterogeneous task weights; off-diagonals separately encode [[label-autocorrelation|label autocorrelation]]. Empirical ECL case study ($T=96$) shows conditional variances vary substantially across future steps, motivating non-uniform diagonals.[^src-qdf]

## Relation to Prior Objectives

| Objective | Per-step / component weights | Autocorrelation handling |
|-----------|------------------------------|---------------------------|
| MSE / DF | uniform | none ($\Sigma=I$) |
| [[fredf\|FreDF]] / Time-o1 | uniform on transformed components | orthogonal / PCA transform (marginal) |
| [[qdf\|QDF]] | learned diagonals of $\Sigma^{-1}$ | learned off-diagonals of $\Sigma^{-1}$ |
| [[source-distdf\|DistDF]] | not framed as multitask weights | joint Wasserstein alignment |

Ablation QDF† (learn diagonals only, zero off-diagonals) already beats DF, showing hetero weights help even without modeling residual correlation; combining both (full QDF) is best.[^src-qdf]

## Related

- [[qdf]], [[quadratic-form-weighted-objective]], [[source-qdf]]
- [[label-autocorrelation]], [[autocorrelation-bias]], [[direct-forecast]]

---

[^src-qdf]: [[source-qdf]]
