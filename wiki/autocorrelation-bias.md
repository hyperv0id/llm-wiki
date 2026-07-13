---
title: "Autocorrelation Bias (Time-Series Forecasting)"
type: concept
tags:
  - time-series-forecasting
  - learning-objective
  - bias
  - mse
created: 2026-07-13
last_updated: 2026-07-13
source_count: 2
confidence: high
status: active
---

## Definition

Autocorrelation bias refers to the bias in the mean squared error (MSE) as an estimator of the conditional negative log-likelihood when the label sequence exhibits autocorrelation. Formally, for label sequence y|x and forecast ŷ|x with conditional covariance Σ|x, the bias is:

Bias = ‖y|x − ŷ|x‖²_Σ⁻¹ − ‖y|x − ŷ|x‖²₂

The bias vanishes only if Σ|x = I — i.e., when future time steps are conditionally independent given the history.[^src-distdf]

## Implications

Standard MSE treats each future step independently, ignoring that y_t depends on y_<t. This oversight renders MSE a biased learning objective for time-series forecasting, hampering model training. A case study on the Traffic dataset shows over 50.3% of conditional correlation matrix entries exceed 0.1, confirming pervasive autocorrelation.[^src-distdf]

## Relationship to Existing Methods

[[source-fredf|FreDF]] first formulated the mismatch between DF's conditional-independence training assumption and [[label-autocorrelation|label autocorrelation]], and proposed frequency-domain alignment as a remedy under the direct forecast paradigm.[^src-fredf]

Later, DistDF reframes the same phenomenon as **autocorrelation bias** of MSE as a likelihood estimator, and argues that likelihood-based transforms still leave residual bias:

- **[[fredf|FreDF]]** uses Fourier transform → guarantees *marginal* decorrelation, not conditional

- **Time-o1** uses PCA → guarantees *marginal* decorrelation, not conditional

Both fail to fully eliminate autocorrelation bias because they cannot guarantee the required *conditional* decorrelation (diagonal Σ|x).[^src-distdf]


## Resolution via DistDF

DistDF bypasses likelihood estimation entirely by directly aligning conditional distributions of forecasts and labels via a joint-distribution Wasserstein discrepancy, avoiding autocorrelation bias at the source.[^src-distdf]

---

[^src-distdf]: [[source-distdf]]
[^src-fredf]: [[source-fredf]]