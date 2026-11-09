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
last_updated: 2026-11-09
source_count: 3
confidence: high
status: active
---

# FreDF

**FreDF (Frequency-enhanced Direct Forecast)** is a plug-and-play training paradigm that improves multi-step time series forecasting by aligning predictions and labels in the frequency domain. Proposed by Hao Wang et al. (ZJU / NTU / PKU et al.), arXiv:2402.02399 (preprint Feb 2024; source file labeled ICLR 2025).[^src-fredf]

## Motivation

Standard [[direct-forecast|direct forecast (DF)]] models minimize step-wise MSE under an implicit conditional independence assumption on the multi-step label sequence. Real labels exhibit [[label-autocorrelation|label autocorrelation]], so DF training is misaligned with the true likelihood (Theorem 3.1). Prior frequency work (e.g., [[fedformer|FEDformer]], [[source-frets|FreTS]]) mostly redesigns *architectures* for *input* autocorrelation; FreDF instead upgrades the *forecast paradigm* for *label* autocorrelation.[^src-fredf]

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

[[distdf|DistDF]] ([[source-distdf]], arXiv:2510.24574, ICLR 2026) treats FreDF as a likelihood-based attempt that achieves only *marginal* frequency decorrelation and therefore leaves residual [[autocorrelation-bias|autocorrelation bias]]; DistDF instead aligns the joint forecast/label distributions via a Wasserstein discrepancy.[^src-distdf]

The theoretical split is the covariance being decorrelated. DistDF's Theorem 3.1 derives the MSE bias from the conditional negative log-likelihood of $y\mid x$, giving $\text{Bias} = \|y_{|x}-\hat y_{|x}\|^2_{\Sigma_{|x}^{-1}} - \|y_{|x}-\hat y_{|x}\|^2_2$, which vanishes only when the *conditional* covariance $\Sigma_{|x}$ equals $I$; Fourier (FreDF) and PCA (Time-o1) transforms guarantee only that the *marginal* covariance $\Sigma$ is diagonal, so $\Sigma_{|x}$ keeps off-diagonal mass and the bias persists.[^src-distdf] The paper's Traffic case study ($T=192$, partial correlation conditioning on history $x$, Appendix C) finds over 50.3% of off-diagonal entries above 0.1 in the raw label correlation matrix, and its Fig. 1b shows the frequency components still carry non-negligible residual conditional correlation after the FreDF transform.[^src-distdf]

Empirically, DistDF reports lower average MSE than FreDF across the four main datasets (Table 1, averaged over $T\in\{96,192,336,720\}$): with TimeBridge, DistDF gives $0.383/0.434/0.172/0.248$ versus FreDF $0.386/0.439/0.175/0.254$ on ETTm1/ETTh1/ECL/Weather; with Fredformer, $0.378/0.430/0.173/0.255$ versus $0.384/0.438/0.179/0.256$.[^src-distdf] [[source-qdf|QDF]] (same author group, ICLR 2026 preprint) similarly argues FreDF/Time-o1 leave residual conditional dependence and equal component weights, and instead learns a quadratic $L_\Sigma$ with adaptive $\Sigma$.[^src-qdf]

## Links

- Source: [[source-fredf]]
- Concepts: [[label-autocorrelation]], [[autocorrelation-bias]], [[direct-forecast]]
- Technique: [[frequency-enhanced-direct-forecast]]
- Related models: [[itransformer]], [[fedformer]], [[source-frets]], [[autoformer]]
- Sibling objectives: [[distdf]], [[joint-distribution-wasserstein-alignment]], [[bures-wasserstein]], [[qdf]], [[source-qdf]], [[source-distdf]]

---

[^src-fredf]: [[source-fredf]]
[^src-distdf]: [[source-distdf]]
[^src-qdf]: [[source-qdf]]

