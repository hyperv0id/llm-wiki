---
title: "Noise Impact Indicator"
type: technique
tags:
  - noise
  - knowledge-distillation
  - regularization
  - spatial-temporal
  - teacher-student
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Noise Impact Indicator

The **Noise Impact Indicator** α̂ is a per-time-series metric proposed in [[rstib-mlp|RSTIB-MLP]] (Chen et al., ICML 2025) to quantify the relative susceptibility of each time series to noise perturbation. It is computed from a pre-trained teacher model's predictions and used to dynamically balance the regularization terms in the RSTIB-MLP learning objective[^src-rstib].

## Definition

Given historical data X^h, target Y, and a trained teacher model f_T:

$$\hat{\alpha}_i = \frac{\exp(D(f_T(A, X^h)_i, Y_i))}{\sum_{j=1}^{N} \exp(D(f_T(A, X^h)_j, Y_j))}, \quad \forall i \in \{1, \ldots, N\}$$

where[^src-rstib]:
- D(·,·) is a distance function (MSE or MAE) measuring prediction error
- A ∈ ℝ^(N×N) is an optional adjacency matrix (teacher-dependent)
- N is the number of time series
- Higher α̂_i → greater prediction error → higher noise susceptibility

## Role in Learning

The indicator is embedded into the [[rstib|RSTIB]] objective as a dynamic weighting factor:

$$\mathcal{L}_{RSTIB-MLP} = \sum_{i=1}^{N} \left[ -\mathcal{L}_{reg}(Y_i^S, \tilde{Y}_i) + (1 + \hat{\alpha}_i)(\lambda_x L_{x,i} + \lambda_y L_{y,i} + \lambda_z L_{z,i}) \right]$$

The (1+α̂_i) multiplier has the effect of[^src-rstib]:
- **Low noise** (small α̂_i): lighter regularization → model uses more raw signal
- **High noise** (large α̂_i): stronger regularization → model compresses more aggressively to filter noise

This dynamic balancing addresses the key limitation of conventional IB: static β multipliers cannot adapt to varying noise levels across different time series or time windows[^src-rstib].

## Properties

- **Model-agnostic**: The teacher f_T can be any model type (STGNN, MLP, etc.). RSTIB-MLP defaults to STGCN[^src-rstib].
- **Window-specific**: α̂_i is computed per time window, adapting to temporal variation in noise patterns[^src-rstib].
- **Related to knowledge distillation**: The teacher model transfers its knowledge of noise patterns to the student RSTIB-MLP, improving feature diversity and representation quality[^src-rstib].

## Significance

The noise impact indicator operationalizes [[rstib|RSTIB]]'s core insight that spatial-temporal forecasting has inherent dynamic relationships: different time series in different windows face different noise levels. Static regularization hyperparameters cannot capture this heterogeneity. α̂_i bridges the gap between the theoretically-grounded RSTIB principle and practical implementation, making it a critical component responsible for balancing informative terms in the objective[^src-rstib].

[^src-rstib]: [[source-rstib-mlp]]
