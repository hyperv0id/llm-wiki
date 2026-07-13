---
title: "FreDF"
type: entity
tags:
  - time-series-forecasting
  - frequency-domain
  - direct-forecast
  - learning-objective
  - plug-and-play
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: high
status: active
---

# FreDF

**FreDF (Frequency-enhanced Direct Forecast)** is a plug-and-play training paradigm that improves multi-step time series forecasting by aligning predictions and labels in the frequency domain. Proposed by Hao Wang et al. (ZJU / NTU / PKU et al.), arXiv:2402.02399 (preprint Feb 2024; source file labeled ICLR 2025).[^src-fredf]

## Motivation

Standard [[direct-forecast|direct forecast (DF)]] models minimize step-wise MSE under an implicit conditional independence assumption on the multi-step label sequence. Real labels exhibit [[label-autocorrelation|label autocorrelation]], so DF training is misaligned with the true likelihood (Theorem 3.1). Prior frequency work (e.g., [[fedformer|FEDformer]], [[frets|FreTS]]) mostly redesigns *architectures* for *input* autocorrelation; FreDF instead upgrades the *forecast paradigm* for *label* autocorrelation.[^src-fredf]

## Method

Given backbone $g$ and history $L$, forecast $\hat Y = g(L)$:

1. Time loss: $L^{(\mathrm{tmp})} = \sum_n \|Y(n)-\hat Y(n)\|_2^2$
2. FFT both sides: $F=\mathcal{F}(Y)$, $\hat F=\mathcal{F}(\hat Y)$
3. Frequency loss: $L^{(\mathrm{feq})} = \sum_n |F(n)-\hat F(n)|$ (sum of complex moduli; not squared)
4. Mixed objective: $L_\alpha = \alpha L^{(\mathrm{feq})} + (1-\alpha) L^{(\mathrm{tmp})}$, $\alpha \in [0,1]$

Because Fourier bases are orthogonal, DML-measured dependence among frequency components is much weaker than among time steps (~3.6% off-diagonals >0.1 vs ~37.5% >0.3 on Weather), better matching DF's independence assumption.[^src-fredf]

See [[frequency-enhanced-direct-forecast]] for the technique-level formulation and variants (2D FFT, polynomial bases).

## Results Snapshot

- Long-term: iTransformer+FreDF leads averaged MSE/MAE on ETT/ECL/Traffic/Weather vs strong baselines (e.g., ETTm1 MSE 0.392 vs iTransformer 0.415).[^src-fredf]
- Short-term M4: improves FreTS on SMAPE/MASE/OWA.[^src-fredf]
- Imputation: improves iTransformer averages over missing ratios.[^src-fredf]
- Generalizes to Autoformer, Transformer, DLinear; orthogonal bases (Fourier/Legendre) outperform non-orthogonal ones.[^src-fredf]

## Relation to Later Work

[[source-distdf|DistDF]] (ICLR 2026) cites FreDF as a likelihood-based attempt that achieves only *marginal* frequency decorrelation and still leaves residual [[autocorrelation-bias|autocorrelation bias]]; DistDF instead aligns joint forecast/label distributions via Wasserstein discrepancy. [[source-qdf|QDF]] (same author group, ICLR 2026 preprint) similarly argues FreDF/Time-o1 leave residual conditional dependence and equal component weights, and instead learns a quadratic $L_\Sigma$ with adaptive $\Sigma$.[^src-fredf]

## Links

- Source: [[source-fredf]]
- Concepts: [[label-autocorrelation]], [[autocorrelation-bias]], [[direct-forecast]]
- Technique: [[frequency-enhanced-direct-forecast]]
- Related models: [[itransformer]], [[fedformer]], [[frets]], [[autoformer]]
- Sibling objectives: [[qdf]], [[source-qdf]], [[source-distdf]]

---

[^src-fredf]: [[source-fredf]]
