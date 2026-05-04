---
title: "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis"
type: source-summary
tags:
  - time-series
  - forecasting
  - periodicity
  - transformer
  - 2d-variation
  - foundation-model
created: 2026-04-28
last_updated: 2026-05-04
source_count: 2
confidence: high
status: active
---

# TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis

## Overview

Published at ICLR 2023 by Haixu Wu et al. (Tsinghua University). arXiv: [2210.02186](https://arxiv.org/abs/2210.02186). TimesNet proposes a task-general foundation model for time series analysis by transforming 1D time series into 2D tensors based on multi-periodicity. Unlike previous methods that model temporal variations directly in 1D space, TimesNet exploits the observation that real-world time series exhibit **intraperiod-variation** (variations within a single period) and **interperiod-variation** (variations across the same phase of different periods). By reshaping 1D series into 2D tensors, these two variation types become columns and rows respectively, amenable to 2D convolutional kernels adapted from computer vision.

## Key Method

TimesNet is composed of stacked **TimesBlocks** in a residual configuration. Each TimesBlock operates in two stages[^src-timesnet]:

### 1. Transforming 1D-Variations into 2D-Variations

Given a length-$T$ time series $\mathbf{X}_{\text{1D}} \in \mathbb{R}^{T \times C}$ with $C$ recorded variates, TimesNet first discovers the most significant periods via FFT[^src-timesnet]:

$$\mathbf{A} = \text{Avg}\left(\text{Amp}\left(\text{FFT}(\mathbf{X}_{\text{1D}})\right)\right), \quad \{f_1, \dots, f_k\} = \underset{f_* \in \{1,\dots,[T/2]\}}{\arg\text{Topk}}(\mathbf{A}), \quad p_i = \left\lceil \frac{T}{f_i} \right\rceil$$

where $\mathbf{A} \in \mathbb{R}^T$ is the amplitude of each frequency averaged across $C$ dimensions. The top-$k$ frequencies $\{f_1, \dots, f_k\}$ correspond to $k$ period lengths $\{p_1, \dots, p_k\}$. Frequencies are restricted to $\{1, \dots, [T/2]\}$ due to frequency-domain conjugacy — this also avoids noise from meaningless high frequencies[^src-timesnet].

For each discovered period $p_i$, the 1D series is padded (zero-padding along temporal dimension) and reshaped into a 2D tensor[^src-timesnet]:

$$\mathbf{X}_i^{\text{2D}} = \text{Reshape}_{p_i, f_i}\left(\text{Padding}(\mathbf{X}_{\text{1D}})\right), \quad i \in \{1, \dots, k\}$$

where $\mathbf{X}_i^{\text{2D}} \in \mathbb{R}^{p_i \times f_i \times C}$. Columns represent **intraperiod-variation** (adjacent time points within a period), and rows represent **interperiod-variation** (same-phase points across different periods). This transformation introduces two types of localities — adjacent time points and adjacent periods — making the tensor amenable to 2D convolutional kernels[^src-timesnet].

### 2. TimesBlock: 2D Representation Learning + Adaptive Aggregation

For the $l$-th TimesBlock with input $\mathbf{X}_{1D}^{l-1} \in \mathbb{R}^{T \times d_{\text{model}}}$[^src-timesnet]:

1. **Period discovery** — Apply FFT to discover $k$ periods adaptively from deep features: $\mathbf{A}^{l-1}, \{f_1, \dots, f_k\}, \{p_1, \dots, p_k\} = \text{Period}(\mathbf{X}_{1D}^{l-1})$.

2. **1D→2D Reshape** — Reshape into $k$ separate 2D tensors: $\mathbf{X}_{i}^{\text{2D}} = \text{Reshape}_{p_i, f_i}(\text{Padding}(\mathbf{X}_{1D}^{l-1}))$.

3. **Inception Block** — Process each 2D tensor via a parameter-efficient **Inception block** (multi-scale 2D convolutions with kernels of varying sizes): $\widehat{\mathbf{X}}_i^{\text{2D}} = \text{Inception}(\mathbf{X}_i^{\text{2D}})$. The Inception block captures both intraperiod and interperiod multi-scale patterns simultaneously.

4. **2D→1D Back** — Reshape processed 2D features back to 1D: $\widehat{\mathbf{X}}_i^{\text{1D}} = \text{Trunc}\left(\text{Reshape}_{1, (p_i \times f_i)}(\widehat{\mathbf{X}}_i^{\text{2D}})\right)$.

5. **Adaptive Aggregation** — Fuse the $k$ representations via softmax-weighted sum based on FFT amplitude values: $\widehat{\mathbf{A}}^{l-1}_{f_1}, \dots, \widehat{\mathbf{A}}^{l-1}_{f_k} = \text{Softmax}(A^{l-1}_{f_1}, \dots, A^{l-1}_{f_k})$. Stronger spectral energy → higher weight.

6. **Residual connection**: $\mathbf{X}_{1D}^{l} = \text{TimesBlock}(\mathbf{X}_{1D}^{l-1}) + \mathbf{X}_{1D}^{l-1}$.

A key architectural property is **generality in 2D vision backbones**: the Inception block can be replaced with any vision architecture (ResNet, ResNeXt, ConvNeXt, Swin Transformer, etc.). The paper shows that replacing Inception with stronger vision backbones consistently improves performance, bridging time series analysis with the broader computer vision community[^src-timesnet]. The parameter-sharing design also makes the model size invariant to the selection of $k$.

## Results

TimesNet achieves **consistent state-of-the-art across five mainstream tasks**[^src-timesnet]:

### Long-term Forecasting
Evaluated on ETT (4 subsets: ETTh1, ETTh2, ETTm1, ETTm2), Electricity, Traffic, Weather, Exchange, and ILI. Past sequence length: 96 (36 for ILI). Prediction lengths: {96, 192, 336, 720} ({24, 36, 48, 60} for ILI). TimesNet achieves SOTA in >80% of cases, outperforming 15+ baselines: [[informer|Informer]], Autoformer, FEDformer, DLinear, ETSformer, LightTS, Non-stationary Transformer, Pyraformer, LogTrans, Reformer, etc. It is the only model that consistently ranks top-2 across all datasets[^src-timesnet].

### Short-term Forecasting (M4)
Evaluated on M4 with ~100K univariate series across Yearly, Quarterly, Monthly, Weekly, Daily, and Hourly frequencies (prediction lengths 6–48). TimesNet achieves SMAPE 11.829, surpassing specialized models N-BEATS (11.851), N-HiTS (11.927), and all Transformer-based models. The diverse data sources make M4 particularly challenging — TimesNet's periodicity-aware design handles this heterogeneity well[^src-timesnet].

### Imputation
On ETT, Electricity, and Weather datasets with random masking ratios {12.5%, 25%, 37.5%, 50%}, TimesNet consistently achieves SOTA MSE across all ratios, demonstrating strong capacity to discover underlying patterns from irregular, partially observed data[^src-timesnet].

### Classification
On 10 multivariate UEA subsets (gesture, action, audio recognition, medical diagnosis), TimesNet achieves **73.6% average accuracy**, surpassing Rocket (72.5%), Flowformer (73.0%), and all Transformer-based models. Notably, MLP-based DLinear fails dramatically (67.5%) — suitable for autoregressive tasks but degenerates in high-level representation learning. TimesNet's 2D-variation modeling benefits hierarchical representation learning[^src-timesnet].

### Anomaly Detection
On five widely-used benchmarks (SMD, MSL, SMAP, SWaT, PSM) using reconstruction error as the shared anomaly criterion, TimesNet achieves best average F1-score. The vanilla Transformer performs poorly (avg F1 76.88%) because attention can be distracted by dominant normal patterns. Periodicity-aware models (TimesNet, FEDformer, Autoformer) all perform well — periodicity analysis implicitly highlights deviations that violate expected patterns[^src-timesnet].

TimesNet also achieves **better efficiency** than most Transformer-based models, with fewer parameters and faster training iterations due to the compact multi-scale 2D inception design[^src-timesnet].

## Critique

**Strengths**: (1) The 1D→2D transformation is conceptually elegant — it decomposes complex temporal variations into orthogonal axes that map naturally to 2D convolutional kernels, exploiting the mature vision backbone ecosystem. (2) Multi-periodicity discovery via FFT is adaptive and modular (one per period), and adaptive aggregation via softmax-weighted amplitude avoids manual period selection. (3) Consistent SOTA across five diverse tasks (forecasting, imputation, classification, anomaly detection) on 25+ datasets is a strong validation of generality.

**Limitations**: (1) FFT-based period discovery assumes reasonably clear periodic structure — time series with weak/no periodicity default to "infinite period" where the 2D tensor becomes a 1×T strip dominated by intraperiod-variation only, potentially losing the benefit of 2D kernels. The paper acknowledges this but does not deeply analyze failure modes. (2) The Inception block, while parameter-efficient, uses relatively shallow convolutions — deeper vision backbones (ConvNeXt, Swin) outperform it, but the paper does not explore scaling behavior. (3) FFT computation introduces $\mathcal{O}(T \log T)$ overhead per layer, though the paper argues this is outweighed by the compact 2D kernel design. (4) No analysis of non-periodic data scenarios, long-horizon extrapolation, or robustness to distribution shifts.

**Comparison to related work**: TimesNet's 2D reshaping is orthogonal to [[informer|Informer]]'s sparse attention and [[autoformer|Autoformer]]'s autocorrelation — while those operate in 1D, TimesNet reformulates the representation space itself. [[hyperd|HyperD]]'s decoupling of short/long-term periodicity into separate pathways is a specialized alternative to TimesNet's shared 2D convolution over multiple discovered periods. [[cyclenet|CycleNet]]'s explicit cycle parameter learning and residual forecasting is complementary — one could imagine combining TimesNet's 2D representation with CycleNet's learned cycles.

[^src-timesnet]: [[source-timesnet]]
[^src-zhou-informer-2021]: [[source-zhou-informer-2021]]
