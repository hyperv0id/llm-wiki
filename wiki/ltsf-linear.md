---
title: "LTSF-Linear"
type: entity
tags:
  - time-series
  - forecasting
  - linear-model
  - baseline
  - LTSF
  - transformer-critique
created: 2026-05-30
last_updated: 2026-05-30
source_count: 1
confidence: high
status: active
---

# LTSF-Linear

**LTSF-Linear** is a set of embarrassingly simple one-layer linear models introduced by Zeng et al. (2022) as a baseline for long-term time series forecasting (LTSF). Despite their simplicity, LTSF-Linear models **outperform all existing Transformer-based LTSF models** on nine benchmarks, often by 20%–50%, raising fundamental questions about the effectiveness of Transformer architectures for time series forecasting[^src-zeng-2022-are-transformers-effective].

## Core Formulation

The basic formulation directly regresses historical time series for future prediction via a weighted sum: $\hat{X}_i = W X_i$, where $W \in \mathbb{R}^{T \times L}$ is a linear layer along the temporal axis, and $\hat{X}_i$ and $X_i$ are the prediction and input for each $i$th variate. LTSF-Linear shares weights across different variates and does not model any spatial correlations[^src-zeng-2022-are-transformers-effective].

## Three Variants

### Vanilla Linear
A single one-layer linear model: $\hat{X}_i = W X_i$. Simplest DMS forecasting via temporal linear layer.

### DLinear (Decomposition-Linear)
Combines the decomposition scheme from [[autoformer|Autoformer]] with linear layers:
1. Decompose raw input into **trend** (moving average kernel, size=25) and **seasonal** (remainder) components
2. Apply separate one-layer linear layers to each component
3. Sum the two features for final prediction

DLinear explicitly handles trend, enhancing vanilla linear when there is a clear trend in the data. The decomposition follows [[autoformer|Autoformer]] and [[fedformer|FEDformer]]'s approach[^src-zeng-2022-are-transformers-effective].

### NLinear (Normalization-Linear)
To handle **distribution shift** between training and testing data:
1. Subtract the last value of the input sequence: $X_i' = X_i - X_i^{L}$
2. Pass through linear layer: $\hat{X}_i' = W X_i'$
3. Add back the subtracted value: $\hat{X}_i = \hat{X}_i' + X_i^{L}$

This simple normalization avoids large errors when model predictions fall outside the distribution of true values. NLinear anticipates later [[instance-normalization|RevIN-style]] approaches[^src-zeng-2022-are-transformers-effective].

## Compelling Characteristics

- **O(1) maximum signal traversing path length**: Shorter path means better dependency capture, enabling both short-range and long-range temporal relations.
- **High efficiency**: 0.04G MACs, 139.7K parameters (DLinear: 2× TL), 0.4ms inference — orders of magnitude cheaper than Transformers.
- **Interpretability**: Weight visualization reveals periodicity patterns (e.g., Traffic weights show daily 24-step and weekly 168-step periodicity) and trend sensitivity (closer time steps receive higher weights on Exchange-Rate).
- **Easy-to-use**: No hyperparameter tuning required[^src-zeng-2022-are-transformers-effective].

## Performance Highlights

On nine widely-used benchmarks (ETTh1/h2, ETTm1/m2, Traffic, Electricity, Exchange-Rate, Weather, ILI):
- LTSF-Linear outperforms SOTA [[fedformer|FEDformer]] in most cases by 20%–50% on multivariate forecasting
- Even naive Repeat outperforms all Transformers on Exchange-Rate by ~45%
- NLinear excels on datasets with distribution shift (ETTh1/h2, ILI)
- DLinear excels on datasets with clear trend/seasonality (Traffic, Weather)[^src-zeng-2022-are-transformers-effective]

## Why Linear Beats Transformers

The paper argues that long-term forecasting depends primarily on capturing **trend and periodicity** — information that linear models naturally extract. Self-attention, being permutation-invariant, inevitably loses temporal ordering despite positional encoding. Most Transformers overfit temporal noises instead of extracting temporal relations from longer sequences[^src-zeng-2022-are-transformers-effective].

## Limitations

- Cannot capture temporal dynamics from change points
- Limited model capacity — one linear layer
- Merely serves as a strong baseline, not a final solution

## Connections

- Directly challenges [[informer]], [[autoformer]], [[fedformer]] on [[lstf|LSTF]] benchmarks
- DLinear's decomposition inherits from [[autoformer|Autoformer]]
- NLinear's distribution shift handling relates to [[instance-normalization|RevIN/Instance Normalization]]
- Supports the value of [[frequency-enhanced-block|domain-specific inductive bias]] (FEDformer's competitiveness on ETTh1)

[^src-zeng-2022-are-transformers-effective]: [[source-zeng-2022-are-transformers-effective]]
