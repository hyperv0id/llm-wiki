---
title: "Autoformer"
type: entity
tags:
  - time-series
  - forecasting
  - periodicity
  - transformer
  - decomposition
  - NeurIPS-2021
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# Autoformer

**Autoformer** is a pioneering Transformer architecture for long-term time series forecasting proposed by Haixu Wu et al. (Tsinghua University), published at NeurIPS 2021. It renovates the Transformer by introducing a progressive decomposition architecture and an Auto-Correlation mechanism that operates at the sub-series level rather than the point level, achieving O(L log L) complexity[^src-autoformer].

## Overview

Prior Transformer models faced two limitations: intricate temporal patterns make point-wise self-attention unreliable for long-range dependencies, and sparse attention variants sacrifice information utilization for efficiency. Autoformer addresses both by replacing self-attention with series-wise Auto-Correlation and embedding decomposition as an inner block, enabling the model to alternately decompose and refine hidden variables[^src-autoformer].

## Key Innovation

Autoformer introduces two major innovations[^src-autoformer]:

1. **Progressive Decomposition Architecture** — Unlike preprocessing decomposition (separating trend/seasonality before modeling), series decomposition is embedded as an inner block within the deep network. Moving average extracts the trend-cyclical component, and the residual becomes the seasonal component, enabling trend extraction even from predicted future states.

2. **Auto-Correlation Mechanism** — Inspired by stochastic process theory, it replaces self-attention entirely. Autocorrelation R(τ) is computed via FFT (Wiener-Khinchin theorem) for all time delays τ. Top-k delays are selected, the value series is rolled by each delay to align sub-series at the same phase position across periods, and fused via softmax-weighted sum based on autocorrelation confidences.

## Architecture

Autoformer follows an encoder-decoder structure where both contain series decomposition blocks and Auto-Correlation modules. The encoder focuses on seasonal parts while the decoder refines trend components. Auto-Correlation achieves O(L log L) complexity while expanding aggregation from point-wise to sub-series level, breaking the bottleneck of sparse attention. The decomposition architecture generalizes to other Transformer models (Transformer, Informer, LogTrans, Reformer), consistently improving performance[^src-autoformer].

## Performance

Autoformer achieves 38% relative MSE improvement over prior state-of-the-art on six benchmarks (ETT, Electricity, Exchange, Traffic, Weather, ILI). Under input-96-predict-336, MSE is reduced by 74% on ETT, 18% on Electricity, 61% on Exchange, and 15% on Traffic. Auto-Correlation also shows superior memory efficiency, handling input-336-predict-1440 where full attention runs out of memory[^src-autoformer].

## Connections

- **[[fedformer]]** — FEDformer builds directly on Autoformer's decomposition philosophy, extending it with learnable MOEDecomp and frequency-domain attention for further improvements.
- **[[timesnet]]** — TimesNet extends Autoformer's periodicity focus by transforming 1D series into 2D tensors for multi-periodicity modeling, while Autoformer treats periodicity through autocorrelation at the seasonal-part level.
- **[[source-dualformer|Dualformer]]** — Dualformer's frequency-branch autocorrelation attention is directly inspired by Autoformer's Wiener-Khinchin theorem approach, extending it with hierarchical frequency sampling.

[^src-autoformer]: [[source-autoformer]]
