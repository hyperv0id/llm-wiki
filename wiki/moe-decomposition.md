---
title: "Mixture of Experts Decomposition (MOEDecomp)"
type: technique
tags:
  - time-series
  - decomposition
  - seasonal-trend
  - mixture-of-experts
  - transformer
  - trend-extraction
  - ICML-2022
created: 2026-05-04
last_updated: 2026-05-04
source_count: 1
confidence: high
status: active
---

# Mixture of Experts Decomposition (MOEDecomp)

**Mixture of Experts Decomposition (MOEDecomp)** is a learnable seasonal-trend decomposition block introduced in [[fedformer|FEDformer]] (ICML 2022). It replaces the fixed-window moving average used in prior decomposition-based models ([[autoformer|Autoformer]], [[informer|Informer]]) with a data-adaptive **mixture of multiple average filters** of different sizes, combined via learned softmax weights[^src-fedformer].

## Motivation

Fixed-window average pooling for trend extraction has a fundamental limitation[^src-fedformer]:

- **Too small a window** → trend captures high-frequency noise, seasonal components leak into trend
- **Too large a window** → trend oversmoothed, important local trend changes lost
- **Fixed size** → cannot adapt to data with **complex periodic patterns** coupled with trend

Real-world time series (energy, traffic, weather) often have multiple overlapping periodicities. A single moving average cannot disentangle trend from all of them simultaneously.

## Formulation

MOEDecomp contains two components[^src-fedformer]:

1. **A set of expert average filters** $\mathcal{F}(\cdot) = \{F_1, F_2, \dots, F_K\}$ with different kernel sizes
2. **Data-dependent mixing weights** produced by a learned linear function: $\text{Softmax}(L(x))$, where $L(x) \in \mathbb{R}^K$

The trend is extracted as:

$$X_{\text{trend}} = \text{Softmax}(L(x)) \cdot \mathcal{F}(x) = \sum_{k=1}^K w_k(x) \cdot F_k(x)$$

where $w_k(x) = \frac{\exp(L_k(x))}{\sum_j \exp(L_j(x))}$ are the data-dependent weights.

The seasonal component is the residual: $X_{\text{seasonal}} = x - X_{\text{trend}}$.

### Key Property: Input-Adaptive Weighting

Unlike a fixed ensemble, the mixing weights $w_k(x)$ depend on the **actual input data** $x$. This means[^src-fedformer]:
- Periods with strong trends → larger windows get higher weight
- Periods with rapid trend changes → smaller windows get higher weight
- Different channels/variables can receive different filter mixtures

## Role in FEDformer Architecture

MOEDecomp is used **at every decomposition point** in both encoder and decoder. It follows each [[frequency-enhanced-block|FEB]] and [[frequency-enhanced-attention|FEA]] block, as well as each FeedForward layer[^src-fedformer]:

**Encoder:**
$$\begin{aligned}S_{en}^{l,1}, \_ &= \text{MOEDecomp}(\text{FEB}(X_{en}^{l-1}) + X_{en}^{l-1}) \\ S_{en}^{l,2}, \_ &= \text{MOEDecomp}(\text{FFN}(S_{en}^{l,1}) + S_{en}^{l,1})\end{aligned}$$

**Decoder (3 per layer):**
$$\begin{aligned}S_{de}^{l,1}, T_{de}^{l,1} &= \text{MOEDecomp}(\text{FEB}(X_{de}^{l-1}) + X_{de}^{l-1}) \\ S_{de}^{l,2}, T_{de}^{l,2} &= \text{MOEDecomp}(\text{FEA}(S_{de}^{l,1}, X_{en}^N) + S_{de}^{l,1}) \\ S_{de}^{l,3}, T_{de}^{l,3} &= \text{MOEDecomp}(\text{FFN}(S_{de}^{l,2}) + S_{de}^{l,2})\end{aligned}$$

The trend components are accumulated across layers: $T_{de}^l = T_{de}^{l-1} + \sum_{i=1}^3 W_{l,i} \cdot T_{de}^{l,i}$.

This progressive decomposition ensures that at each layer, the model refines its understanding of what is "trend" vs. "seasonal," with the accumulated trend $T_{de}^M$ contributing directly to the final prediction[^src-fedformer].

## Empirical Effectiveness

On ETT and Weather datasets, MOEDecomp brings an **average of 2.96% improvement** over a single fixed-window decomposition scheme[^src-fedformer].

Additionally, MOEDecomp contributes to FEDformer's key achievement: being the **only model** where the KS test cannot reject the null hypothesis ($P > 0.01$) that predicted and input sequences come from the same distribution. The adaptive trend extraction preserves distributional properties better than fixed decomposition[^src-fedformer].

## Comparison with Other Decomposition Methods

| Method | Trend Extraction | Adaptivity | Complexity |
|:-------|:-----------------|:-----------|:-----------|
| **MOEDecomp** | Mixture of multiple average filters | **Data-dependent weights** | $O(K \cdot L)$ |
| Autoformer | Fixed moving average | None | $O(L)$ |
| DLinear | Single moving average kernel | None | $O(L)$ |
| STL (classic) | LOESS smoothing | None (manual parameter) | $O(L^2)$ or higher |
| CycleNet RCF | Learnable recurrent cycles | Learned globally (not input-adaptive) | $O(W \cdot D)$ |

**Note**: "Adaptivity" here means adapting to the **specific input instance** at inference time. CycleNet's learnable cycles are trained globally but fixed at inference.

## Connections

- **[[fedformer|FEDformer]]** — the model that introduced MOEDecomp (ICML 2022)
- **[[frequency-enhanced-block|FEB]]** — precedes MOEDecomp in the FEDformer pipeline
- **[[frequency-enhanced-attention|FEA]]** — precedes MOEDecomp in the decoder
- **[[autoformer|Autoformer]]** — uses fixed moving average decomposition; MOEDecomp makes this learnable
- **[[cyclenet|CycleNet]]** — uses [[residual-cycle-forecasting|RCF]] with learnable cycles for periodicity modeling, a different decomposition philosophy
- **[[mixture-of-experts|MoE]]** — the general mixture-of-experts concept applied to decomposition filters
- **[[seasonal-trend-decomposition]]** — the broader class of methods MOEDecomp belongs to

[^src-fedformer]: [[source-fedformer]]