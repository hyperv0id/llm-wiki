---
title: "Long Sequence Time-Series Forecasting (LSTF)"
type: concept
tags:
  - time-series
  - forecasting
  - long-sequence
  - transformer
created: 2026-05-04
last_updated: 2026-05-31
source_count: 4
confidence: high
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
| **[[crossformer|Crossformer]]** | 2023 | $O(DL^2_\text{seg})$ | DSW 2D embedding + TSA cross-dimension attention + HED (ICLR) |
| **[[patchtst|PatchTST]]** | 2023 | $O((L/S)^2)$ | Patch tokenization + Channel Independence + self-supervised (ICLR) |
| **[[itransformer|iTransformer]]** | 2024 | $O(N^2)$ per layer | Inverted dimensions: attention on variates, FFN on time (ICLR) |
| **[[sparsetsf|SparseTSF]]** | 2025 | Extreme compression | Cross-period sparse forecasting with <1k parameters (ICML/TPAMI) |

The progression shows a trend from **efficiency-first** (Informer: reduce complexity) → **structure-first** (Autoformer: embed decomposition) → **domain-specific** (FEDformer/FreTS: frequency domain) → **cross-dimension** (Crossformer: explicit variable interaction) → **tokenization rethink** (PatchTST: patch + CI, proving Transformer can beat linear models) → **architecture rethinking** (iTransformer: invert dimensions without modifying components) → **extreme compression** (SparseTSF: sub-1k-parameter models).

## iTransformer 对 LSTF 的突破

传统 Transformer 在 LSTF 中存在一个长期痛点：**随回看窗口增长性能不提升**——注意力在更长输入上分散。iTransformer 解决了这一问题：由于 FFN 作用于时间维度（等价共享线性预测器），扩展回看窗口带来更多历史信息，性能持续提升，与统计方法的理论期望一致[^src-itransformer]。这使得 iTransformer 成为 LSTF 场景下更合理的 Transformer backbone。

## Linear Model Challenge

Zeng et al. (2022) fundamentally challenged the Transformer-based LSTF paradigm with [[ltsf-linear|LTSF-Linear]] — embarrassingly simple one-layer linear models that outperform all existing Transformer-based LTSF models on nine benchmarks by 20%–50%. Key findings: (1) Transformers fail to exploit longer input sequences (performance deteriorates with increasing look-back window), while LTSF-Linear improves significantly; (2) self-attention is permutation-invariant and inevitably loses temporal ordering; (3) shuffling input barely affects Transformer performance on Exchange-Rate, confirming limited temporal relation extraction. The paper argues that long-term forecasting depends primarily on capturing trend and periodicity — information that linear models naturally extract[^src-zeng-2022-are-transformers-effective].

**PatchTST 的回应**：PatchTST (ICLR 2023) 通过 patching + channel independence 证明正确设计的 Transformer 可以超越 DLinear，且是唯一随 look-back window 增大持续降低 MSE 的 Transformer 模型[^src-patchtst]。

## Relationship to Other Forecasting Settings

- **Short-term forecasting**: Usually ≤48 steps; can use simpler models (ARIMA, LSTMs, vanilla MLPs).
- **LSTF**: ≥96 input, ≥96 output; requires efficient architectures due to complexity scaling.
- **Multi-horizon forecasting**: Predicts multiple future horizons simultaneously — LSTF is a specific instantiation with long horizons.

[^src-zhou-informer-2021]: [[source-zhou-informer-2021]]
[^src-zeng-2022-are-transformers-effective]: [[source-zeng-2022-are-transformers-effective]]
[^src-itransformer]: [[source-itransformer]]
[^src-patchtst]: [[source-patchtst]]