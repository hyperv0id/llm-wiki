---
title: "RSTIB-MLP"
type: entity
tags:
  - time-series
  - spatial-temporal
  - information-bottleneck
  - mlp
  - robustness
  - traffic-forecasting
  - icml-2025
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# RSTIB-MLP

**RSTIB-MLP** (Robust Spatial-Temporal Information Bottleneck MLP) is an MLP-based spatial-temporal forecasting model that achieves robust prediction under noise perturbation while maintaining computational efficiency. Proposed by Chen et al. (ICML 2025), it addresses the dual noise effect in STF—where noise harms both input and target ends via the sliding window mechanism—by instantiating the RSTIB principle on pure MLP networks augmented with a knowledge distillation module[^src-rstib].

## Core Architecture

### RSTIB Principle Instantiation

RSTIB-MLP instantiates the [[rstib|RSTIB principle]] through three mechanics, each with analytical upper/lower bounds[^src-rstib]:

1. **Input/Target Regularization**: I(X;Y;Z) is minimized by imposing KL divergence from parameterized Gaussian distributions to unit Gaussian N(0,1) on both reparameterized input X̃ and target Ỹ. This has closed-form analytical solutions[^src-rstib].

2. **Representation Regularization**: I(Z;X,Y) is minimized by reparameterizing Z ~ N(μ_z, σ_z²) from reparameterized input X̃, then imposing KL divergence to N(0,1) on Z[^src-rstib].

3. **Prediction Objective**: I(Z;Ỹ) is maximized via a standard regression loss (MAE) between predictions Y^S and reparameterized target Ỹ, serving as a variational lower bound derived without the Z–X–Y Markov restriction[^src-rstib].

All regularization terms have the same closed form: KL(N(μ,σ²)||N(0,1)) = ½(−log σ² + μ² + σ² − 1)[^src-rstib].

### Knowledge Distillation Module

A teacher model f_T (model-agnostic; STGCN by default) generates predictions Y^T, from which per-series noise impact indicators α̂_i are computed:

$$\hat{\alpha}_i = \frac{\exp(D(f_T(A, X^h)_i, Y_i))}{\sum_{j=1}^{N} \exp(D(f_T(A, X^h)_j, Y_j))}$$

where D is a distance function (MSE or MAE). Higher α̂_i → greater noise susceptibility → stronger regularization. These indicators dynamically balance λ_x, λ_y, λ_z multipliers in the objective[^src-rstib].

### Spatial-Temporal Prompts

Input X^h is augmented with learnable spatial-temporal prompts (extensions of [[stid|STID]]'s spatial-temporal identities): static spatial prompt E^(α), dynamic transitional prompt E^(β), time-of-day prompt E^(ToD), and day-of-week prompt E^(DoW), concatenated via FC layers[^src-rstib].

### Final Objective

$$\mathcal{L}_{RSTIB-MLP} = \sum_{i=1}^{N} \left[ -\mathcal{L}_{reg}(Y_i^S, \tilde{Y}_i) + (1 + \hat{\alpha}_i)(\lambda_x L_{x,i} + \lambda_y L_{y,i} + \lambda_z L_{z,i}) \right]$$

The (1+α̂_i) factor dynamically intensifies regularization for noise-vulnerable time series[^src-rstib].

## Performance

### Robustness (noise ratios 0%–50%, six datasets)

RSTIB-MLP achieves **best or second-best** MAE/RMSE/MAPE in 90%+ of noise scenarios across PEMS04/07/08, LargeST(SD), Weather2K-R, and Electricity, compared to 10 baselines: STID, GWN, TrendGCN, STExplainer, STExplainer-CGIB, STGKD, BiTGraph, STC-Dropout, STG-NCDE, FreTS[^src-rstib].

### Efficiency

On PEMS04: ~180 seconds/epoch vs 600–2000+ seconds for STGNN-based methods. Slightly slower than the pure MLP baseline STID (~150 seconds/epoch) but with dramatically better robustness[^src-rstib].

### Ablation

Removing IB entirely, using vanilla IB instead of RSTIB, or removing KD all cause significant degradation under noise, with vanilla IB sometimes performing worse than no IB (due to ignoring target-end noise)[^src-rstib].

### Feature Variance

KD module significantly boosts feature variance (diversity of learned representations), a critical factor against feature collapse under noise. The dual noise effect causes faster variance degradation than single-end noise[^src-rstib].

## Connections

- **[[rstib|RSTIB Principle]]**: Theoretical foundation, generalizing RGIB and lifting Z–X–Y Markov assumption[^src-rstib]
- **[[noise-impact-indicator|Noise Impact Indicator]]**: Per-series α̂_i computed from teacher model predictions, enabling dynamic regularization[^src-rstib]
- **[[stid|STID]]**: RSTIB-MLP uses spatial-temporal prompts evolved from STID's spatial-temporal identities[^src-rstib]
- **[[ltsf-linear]]**: Both demonstrate MLPs' viability for time series; RSTIB-MLP adds robustness via IB-guided regularization[^src-rstib]
- **[[timemixer]]**: Both are MLP-based time series models; RSTIB-MLP trades TimeMixer's multi-scale mixing focus for explicit noise robustness[^src-rstib]
- **[[information-bottleneck]]**: Extends IB/[[dvib|DVIB]]/[[gib|GIB]]/RGIB lineage to spatial-temporal forecasting with dual-noise handling[^src-rstib]
- **[[frets|FreTS]]**: FreTS uses frequency-domain MLPs for energy compaction (implicit noise filtering); RSTIB-MLP uses information theory for explicit noise regularization. RSTIB-MLP outperforms FreTS under most noise conditions[^src-rstib]

[^src-rstib]: [[source-rstib-mlp]]
