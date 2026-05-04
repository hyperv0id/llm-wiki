---
title: "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting"
type: source-summary
tags:
  - time-series
  - forecasting
  - periodicity
  - transformer
  - auto-correlation
  - decomposition
  - series-wise
created: 2026-04-28
last_updated: 2026-05-04
source_count: 1
confidence: medium
status: active
---

# Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting

## Overview

Published at NeurIPS 2021 by Haixu Wu et al. (Tsinghua University), Autoformer is a pioneering work that renovates the Transformer architecture for long-term time series forecasting. It addresses two key limitations of prior Transformer-based models: (1) intricate temporal patterns make point-wise self-attention unreliable for discovering long-range dependencies, and (2) sparse self-attention variants sacrifice information utilization for efficiency. Autoformer introduces a **progressive decomposition architecture** and an **Auto-Correlation mechanism** that operates at the sub-series level rather than the point level.

## Key Method

### Series Decomposition Block
Unlike traditional pre-processing decomposition (separating trend/seasonality before modeling), Autoformer embeds **series decomposition as an inner block** within the deep network. Moving average extracts the trend-cyclical component, and the residual becomes the seasonal component. This progressive decomposition allows the model to alternately decompose and refine intermediate hidden variables throughout the encoder-decoder, extracting trend information even from predicted future states.

### Auto-Correlation Mechanism
Inspired by stochastic process theory, Auto-Correlation replaces self-attention entirely. It works in three steps:
1. **Period discovery** — Autocorrelation R(τ) is computed via FFT (Wiener–Khinchin theorem) for all time delays τ in O(L log L) time, revealing period-based dependencies.
2. **Time delay aggregation** — Top-k delays τ₁, …, τₖ are selected. The value series V is **rolled** by each delay, aligning sub-series at the same phase position across periods.
3. **Weighted fusion** — Rolled sub-series are aggregated via softmax-weighted sum based on autocorrelation confidences.

This achieves **O(L log L) complexity** while expanding aggregation from point-wise to sub-series level, breaking the information utilization bottleneck of sparse attention.

## Results

Autoformer achieves **38% relative MSE improvement** over previous state-of-the-art on six benchmarks (ETT, Electricity, Exchange, Traffic, Weather, ILI) covering energy, traffic, economics, weather, and disease applications. Under input-96-predict-336, it reduces MSE by 74% on ETT, 18% on Electricity, 61% on Exchange, and 15% on Traffic compared to prior SOTA. The decomposition architecture generalizes to other Transformer models (Transformer, Informer, LogTrans, Reformer), consistently improving their performance. Auto-Correlation also shows superior memory efficiency, handling input-336-predict-1440 where full attention runs out of memory.

## Critique

Autoformer's progressive decomposition assumes additive trend + seasonal structure, which may not hold for all real-world series (e.g., multiplicative seasonality or abrupt regime changes). The FFT-based autocorrelation computation, while efficient, implicitly assumes stationarity; highly non-stationary series may yield unreliable period estimates. Compared to [[informer|Informer]] (AAAI 2021 Best Paper) which pioneered efficient Transformer-based LSTF with ProbSparse attention, Autoformer replaces self-attention entirely rather than making it sparse — a fundamentally different design philosophy[^src-zhou-informer-2021]. Compared to [[hybrid-periodicity-decoupling|Hybrid Periodicity Decoupling]] approaches like [[hyperd|HyperD]], Autoformer treats periodicity at the seasonal-part level rather than separating short-term and long-term cycles. The [[demlp-decoder|fDMLP decoder]] in HyperD was explicitly inspired by Autoformer/DLinear's decomposition philosophy, but extends it to multi-scale fusion. Autoformer also predates and is cited extensively by TimesNet, [[source-timesnet|TimesNet]], and PENGUIN, serving as a foundational reference for periodicity-based time series modeling.


[^src-zhou-informer-2021]: [[source-zhou-informer-2021]]
