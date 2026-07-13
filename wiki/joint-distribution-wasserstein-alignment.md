---
title: "Joint-Distribution Wasserstein Alignment (DistDF)"
type: technique
tags:
  - time-series-forecasting
  - learning-objective
  - optimal-transport
  - wasserstein
  - distribution-alignment
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: high
status: active
---

## Overview

Joint-distribution Wasserstein alignment is a technique for training time-series forecasting models by minimizing the discrepancy between the conditional distributions of forecasts and labels, using a joint-distribution Wasserstein discrepancy as a tractable proxy. Proposed in the DistDF framework (ICLR 2026).

## Motivation

Training forecasting models requires aligning Pŷ|x with Py|x. The standard approach (minimizing conditional negative log-likelihood via MSE) suffers from autocorrelation bias. Directly minimizing a conditional distributional discrepancy is intractable because for any given history x, typically only one label y is observed.[^src-distdf]

## Method

The key insight is to use the **joint-distribution Wasserstein discrepancy** Wp(Px,y, Px,ŷ) as a proxy:

Wp(Px,y, Px,ŷ) ≥ ∫ Wp(Py|x, Pŷ|x) dP(x)

This joint discrepancy:
1. **Upper-bounds** the expected conditional discrepancy (Lemma 3.3)
2. Is **tractable** — empirical samples Sx,y and Sx,ŷ can be constructed from the full dataset
3. **Implies alignment** — if Wp(Px,y, Px,ŷ) = 0, then Py|x = Pŷ|x for almost every x (Theorem 3.4)[^src-distdf]

## Practical Loss

Under a Gaussian assumption, the squared W₂ discrepancy reduces to the **Bures-Wasserstein discrepancy**:

BW = ‖µx,y − µx,ŷ‖²₂ + Tr(Σx,y + Σx,ŷ − 2(Σ¹⸝²_{x,y} Σx,ŷ Σ¹⸝²_{x,y})¹⸝²)

The overall DistDF loss combines this with MSE:

LDistDF = γ · BW + (1 − γ) · LMSE

where MSE preserves elementwise correspondences and γ ∈ [0,1] controls the alignment strength. DistDF is model-agnostic and works as a plug-and-play regularization term.[^src-distdf]

## Limitations

The Bures-Wasserstein metric captures only first- and second-order moments. Real-world non-Gaussian data may require higher-order statistics for full characterization.[^src-distdf]

## Related Techniques

- [[source-fredf|FreDF]]: frequency-domain likelihood-based method (marginal decorrelation only)
- [[source-time-o1|Time-o1]]: PCA-based transformed label alignment
- [[source-exost|ExoST]]: exogenous variable modeling for spatiotemporal forecasting

---

[^src-distdf]: [[source-distdf]]