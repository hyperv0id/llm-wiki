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
last_updated: 2026-11-09
source_count: 2
confidence: high
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
| [[distdf\|DistDF]] ([[source-distdf]], arXiv:2510.24574) | not framed as multitask weights; horizon weighting stays implicit | joint Wasserstein alignment on mean and covariance[^src-distdf] |

DistDF does not expose a per-step weight vector: its objective is $\gamma\,BW + (1-\gamma)\,L_{\mathrm{MSE}}$, so any non-uniform horizon weighting is absorbed into the covariance term of the [[bures-wasserstein|Bures–Wasserstein]] discrepancy rather than read off a diagonal.[^src-distdf] In that sense QDF and DistDF both depart from uniform weighting, but only QDF offers an interpretable diagonal.

Ablation QDF† (learn diagonals only, zero off-diagonals) already beats DF, showing hetero weights help even without modeling residual correlation; combining both (full QDF) is best.[^src-qdf]

## Related

- [[qdf]], [[quadratic-form-weighted-objective]], [[source-qdf]]
- [[distdf]], [[joint-distribution-wasserstein-alignment]], [[bures-wasserstein]]
- [[label-autocorrelation]], [[autocorrelation-bias]], [[direct-forecast]]

---

[^src-qdf]: [[source-qdf]]
[^src-distdf]: [[source-distdf]]
