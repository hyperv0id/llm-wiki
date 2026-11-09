---
title: "Frequency-enhanced Direct Forecast"
type: technique
tags:
  - time-series-forecasting
  - frequency-domain
  - direct-forecast
  - fft
  - learning-objective
created: 2026-07-13
last_updated: 2026-11-09
source_count: 2
confidence: high
status: active
---

## Overview

**Frequency-enhanced Direct Forecast (FreDF)** is a training technique that augments the [[direct-forecast|direct forecast]] multi-output head with a frequency-domain alignment loss between forecasts and labels. It is model-agnostic and transform-agnostic: any backbone $g$ and any (preferably orthogonal) basis transform $\mathcal{F}$ can be used. Introduced in [[source-fredf|FreDF]] (arXiv:2402.02399).[^src-fredf]

## Algorithm

```
Input: history L, label Y, model g, mix weight α
Ŷ ← g(L)
L_tmp ← MSE(Y, Ŷ)                         # time-domain DF loss
F, F̂ ← ℱ(Y), ℱ(Ŷ)                        # typically FFT along time
L_feq ← Σ |F − F̂|                          # sum of complex moduli
L ← α · L_feq + (1 − α) · L_tmp
Backprop L through g (FFT is differentiable)
```

Key design choices:[^src-fredf]

- **Modulus loss, not squared**: low-frequency magnitudes dwarf high-frequency ones; squared loss is numerically unstable.
- **α near 1 often best** (e.g., ~0.8 on ETTh1): pure frequency works well; a little time-domain signal can still help.
- **Phase + amplitude both matter**; phase alignment is especially critical.

## Variants

- **Axis of FFT**: 1D along time (label autocorrelation), 1D along variables (cross-variate correlation), or 2D over both (best joint gains).[^src-fredf]
- **Basis sets**: Fourier and Legendre (orthogonal) outperform Chebyshev/Laguerre without proper weighting; orthogonality is the operative property for reducing label dependence.[^src-fredf]
- **Backbones**: demonstrated on iTransformer, DLinear, Autoformer, Transformer, FreTS (short-term).[^src-fredf]

## Sample Efficiency

On ETTh1/ECL learning curves, frequency-domain supervision with ~30% of training data can match full-data time-domain MSE, attributed to more stable frequency patterns under sliding windows (e.g., sine windows share a sparse spectral spike).[^src-fredf]

## Relation to Architecture-Level Frequency Methods

| Method | Where frequency is used | Target dependence |
|--------|-------------------------|-------------------|
| [[fedformer|FEDformer]] / [[frequency-enhanced-block|FEB]] | Attention / blocks | Input encoding |
| [[source-frets|FreTS]] | MLP layers on DFT features | Input encoding |
| **FreDF** | Training loss only | Label sequence under DF |

## Limitations and Follow-ups

Fixed Fourier bases do not adapt to data geometry; PCA-style adaptive orthogonal bases are suggested.[^src-fredf] [[distdf|DistDF]] ([[source-distdf]], arXiv:2510.24574) later argues that frequency/PCA transforms only ensure *marginal* decorrelation and residual [[autocorrelation-bias]] remains: its Theorem 3.1 shows the MSE bias vanishes only when the conditional covariance $\Sigma_{|x}$ is the identity, while Fourier and PCA bases make the marginal covariance $\Sigma$ diagonal at best.[^src-distdf] The Traffic case study ($T=192$, partial correlation conditioning on history) reports over 50.3% of off-diagonal entries above 0.1, and Fig. 1b shows the FreDF frequency components still carry residual conditional correlation; on the objective comparison (Table 1), [[joint-distribution-wasserstein-alignment]] gives average MSE $0.383/0.434/0.172/0.248$ against FreDF's $0.386/0.439/0.175/0.254$ with TimeBridge, and $0.378/0.430/0.173/0.255$ against $0.384/0.438/0.179/0.256$ with Fredformer (ETTm1/ETTh1/ECL/Weather).[^src-distdf]

## Related

- Entity: [[fredf]], [[distdf]]
- Concept: [[label-autocorrelation]], [[autocorrelation-bias]]
- Technique: [[joint-distribution-wasserstein-alignment]]
- Source: [[source-fredf]], [[source-distdf]]
- Frequency-domain low-frequency supervision: [[patch-low-frequency-forecasting]] (LoFT-LLM)

---

[^src-fredf]: [[source-fredf]]
[^src-distdf]: [[source-distdf]]
