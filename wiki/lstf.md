---
title: "Long Sequence Time-Series Forecasting (LSTF)"
type: concept
tags:
  - time-series
  - forecasting
  - long-sequence
  - transformer
created: 2026-05-04
last_updated: 2026-05-04
source_count: 1
confidence: medium
status: active
---

# Long Sequence Time-Series Forecasting (LSTF)

**Long Sequence Time-Series Forecasting (LSTF)** is a forecasting paradigm that requires predicting **long-horizon future values** (e.g., 168, 336, 720 or more steps) from **long historical input sequences** (e.g., 96, 192, or more steps). LSTF was formalized as a distinct problem setting by the [[informer|Informer]] paper (Zhou et al., AAAI 2021 Best Paper)[^src-zhou-informer-2021].

## Motivation

Real-world applications increasingly demand long-horizon predictions:
- **Electricity**: Forecast hourly consumption for the next week (168 steps) to optimize grid scheduling.
- **Weather**: Predict temperature and humidity for the next 14 days (336 steps) for agricultural planning.
- **Traffic**: Anticipate traffic flow over the next 1-3 days (288-864 steps) for congestion management.
- **Finance**: Project exchange rates for the next month (720+ steps) for risk hedging.

Prior to LSTF, most forecasting models were designed for short-term settings (e.g., 12-48 steps). Long-horizon prediction introduces unique challenges: the dependencies to capture are longer-range, the input must be correspondingly longer, and the computational demands grow accordingly.

## Challenges

The LSTF setting exposes three critical bottlenecks in standard Transformer architectures[^src-zhou-informer-2021]:

1. **Quadratic Time Complexity**: Canonical self-attention is $O(L^2)$, making input lengths beyond a few hundred steps computationally prohibitive.
2. **Memory Explosion**: Stacking $J$ transformer layers results in $O(J \cdot L^2)$ memory usage — each layer stores the full attention matrix.
3. **Slow Inference**: Dynamic (autoregressive) decoding requires $L_{pred}$ sequential forward passes, propagating errors and negating the parallel computation advantage.

## Key Models Addressing LSTF

LSTF has driven a sustained research line focused on efficient Transformer architectures, each addressing the bottlenecks in different ways:

| Model | Year | Complexity | Key Innovation |
|-------|------|-----------|----------------|
| **[[informer|Informer]]** | 2021 | $O(L \log L)$ | ProbSparse attention + generative decoder (AAAI Best Paper) |
| **[[autoformer|Autoformer]]** | 2021 | $O(L \log L)$ | Progressive decomposition + Auto-Correlation (NeurIPS) |
| **[[fedformer|FEDformer]]** | 2022 | $O(L)$ | Frequency-enhanced attention via Fourier/Wavelet (ICML) |
| **[[source-frets|FreTS]]** | 2023 | $O(N \log N + L \log L)$ | Frequency-domain MLPs as global convolutions (NeurIPS) |
| **[[sparsetsf|SparseTSF]]** | 2025 | Extreme compression | Cross-period sparse forecasting with <1k parameters (ICML/TPAMI) |

The progression shows a trend from **efficiency-first** (Informer: reduce complexity) → **structure-first** (Autoformer: embed decomposition) → **domain-specific** (FEDformer/FreTS: frequency domain) → **extreme compression** (SparseTSF: sub-1k-parameter models).

## Relationship to Other Forecasting Settings

- **Short-term forecasting**: Usually ≤48 steps; can use simpler models (ARIMA, LSTMs, vanilla MLPs).
- **LSTF**: ≥96 input, ≥96 output; requires efficient architectures due to complexity scaling.
- **Multi-horizon forecasting**: Predicts multiple future horizons simultaneously — LSTF is a specific instantiation with long horizons.

[^src-zhou-informer-2021]: [[source-zhou-informer-2021]]