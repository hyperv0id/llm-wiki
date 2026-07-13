---
title: "DistDF: Time-Series Forecasting Needs Joint-Distribution Wasserstein Alignment"
type: source-summary
tags:
  - time-series-forecasting
  - distribution-alignment
  - optimal-transport
  - wasserstein
  - iclr-2026
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: high
status: active
---

## Summary

**DistDF (Distribution-aware Direct Forecast)** is a training framework for time-series forecasting that replaces likelihood-based objectives with a joint-distribution Wasserstein discrepancy. Published at ICLR 2026 by Hao Wang et al. (Xiaohongshu Inc., Zhejiang University, Peking University).

## Core Arguments

The paper makes three main contributions:

**1. Identifies autocorrelation bias in MSE.** The standard MSE objective for training forecasting models is a biased estimator of the conditional negative log-likelihood when the label sequence exhibits autocorrelation (Theorem 3.1). The bias vanishes only if the conditional covariance Σ|x is the identity matrix — i.e., when future time steps are conditionally independent. A case study on the Traffic dataset shows that over 50.3% of conditional correlation matrix entries exceed 0.1, confirming substantial autocorrelation. Existing likelihood-based methods (FreDF via Fourier transform, Time-o1 via PCA) guarantee only *marginal* decorrelation of components, not the required *conditional* decorrelation, so the bias persists.[^src-distdf]

**2. Proposes joint-distribution Wasserstein discrepancy.** Instead of likelihood estimation, DistDF aligns the conditional distribution of forecasts Pŷ|x with the label distribution Py|x by minimizing a distributional discrepancy. Directly estimating conditional discrepancies from finite observations is intractable (each x typically has only one y). The paper introduces the **joint-distribution Wasserstein discrepancy** Wp(Px,y, Px,ŷ), which provably upper-bounds the expected conditional Wasserstein discrepancy (Lemma 3.3) and is readily estimable from the full dataset. Theorem 3.4 proves that minimizing this joint discrepancy to zero achieves conditional distribution alignment.[^src-distdf]

**3. Practical Bures-Wasserstein implementation.** Under a Gaussian assumption, the squared W2 discrepancy reduces to the closed-form Bures-Wasserstein metric that matches first- and second-order moments (mean and covariance). The overall loss is LDistDF = γ·LDist + (1-γ)·LMSE, combining joint-distribution alignment with pointwise MSE. DistDF is model-agnostic and works as a plug-and-play regularization term.[^src-distdf]

## Experiments

DistDF is evaluated on ETT, ECL, and Weather datasets with forecasting horizons 96/192/336/720. Using TimeBridge and Fredformer as testbed models, DistDF achieves the best MSE/MAE across all settings compared to DF (MSE), Soft-DTW, DILATE, Koopman, FreDF, and Time-o1. Ablation studies confirm that both mean alignment and covariance alignment contribute synergistically. DistDF generalizes across different discrepancy measures (MMD, KL, EMD) and forecasting models (iTransformer, FreTS, TimeBridge, Fredformer), with consistent improvements of 1–4% MSE reduction.[^src-distdf]

## Limitations

The Bures-Wasserstein discrepancy under a Gaussian assumption captures only first- and second-order moments. Real-world data may exhibit non-Gaussian characteristics requiring higher-order statistics for full characterization. Extending to higher-order discrepancies while maintaining computational tractability is a future direction. Sibling work [[source-qdf|QDF]] (same author group) instead remains in the quadratic-likelihood family and learns $\Sigma$ for autocorrelation + heterogeneous step weights rather than switching to OT alignment.[^src-distdf]

## Key Terminology

- **Autocorrelation bias**: the bias in MSE as a negative log-likelihood estimator when label sequences have conditional autocorrelation
- **Joint-distribution Wasserstein discrepancy**: Wp(Px,y, Px,ŷ) as a tractable upper bound on expected conditional Wasserstein discrepancy
- **Bures-Wasserstein discrepancy**: closed-form W2 metric under Gaussian assumption, matching mean and covariance

## Sibling Objectives

- [[source-fredf|FreDF]] — frequency-domain likelihood alignment (marginal decorrelation)
- [[source-qdf|QDF]] — learned quadratic $L_\Sigma$ for autocorrelation + hetero task weights

---

[^src-distdf]: [[source-distdf]]