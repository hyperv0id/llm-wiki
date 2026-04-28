---
title: "FEDformer"
type: entity
tags:
  - time-series
  - forecasting
  - frequency-domain
  - transformer
  - ICML-2022
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# FEDformer

**FEDformer** (Frequency Enhanced Decomposed Transformer) is a Transformer architecture for long-term series forecasting, proposed by Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin (Alibaba Group), published at ICML 2022. It combines seasonal-trend decomposition with frequency-domain attention mechanisms to capture both global time series properties and local detailed structures, while achieving linear complexity[^src-fedformer].

## Overview

Standard Transformer forecasting methods struggle to maintain global time series properties because they make point-wise independent predictions per timestep. FEDformer addresses this by combining seasonal-trend decomposition via a Mixture-of-Experts scheme with frequency-enhanced attention blocks. The paper shows that randomly selecting a fixed number of frequency components yields better representations than keeping only low frequencies[^src-fedformer].

## Key Innovation

FEDformer introduces three novel components[^src-fedformer]:

1. **Frequency Enhanced Block (FEB)** — Substitutes the self-attention block. FEB-f uses Discrete Fourier Transform (DFT) to project inputs into the frequency domain, randomly selects M modes, applies a parameterized kernel, and returns via inverse DFT. FEB-w uses multiwavelet decomposition with Legendre polynomial bases, processing signals recursively across scales.

2. **Frequency Enhanced Attention (FEA)** — Substitutes cross-attention between encoder and decoder. FEA-f converts queries, keys, and values via DFT, applies attention in the frequency domain, and inverts back. FEA-w applies multiwavelet decomposition to q, k, v separately.

3. **Mixture of Experts Decomposition (MOEDecomp)** — Replaces fixed-window average pooling with learnable average filters of different sizes, combined via data-dependent softmax weights, adaptively extracting trend components from complex periodic patterns.

## Architecture

FEDformer follows an encoder-decoder structure. With M randomly selected Fourier modes (default M=64), it achieves O(L) complexity versus O(L²) for standard Transformers. MOEDecomp blocks precede each FEB/FEA block, progressively decomposing seasonal and trend components[^src-fedformer].

## Performance

FEDformer achieves best results across all six benchmarks (ETTm2, Electricity, Exchange, Traffic, Weather, ILI) and horizons. Multivariate MSE is reduced 14.8% vs. Autoformer; univariate by 22.6%. On low-periodicity datasets (Exchange, ILI), improvements exceed 20%. The KS test confirms outputs better match input distributions than competing models[^src-fedformer].

## Connections

- **[[autoformer]]** — Autoformer introduced progressive decomposition and autocorrelation-based attention, both of which directly influenced FEDformer's design. FEDformer extends decomposition by making it learnable via MOEDecomp.
- **[[dualsformer]]** — Dualformer builds on FEDformer's frequency-domain approach but replaces fixed random mode selection with input-adaptive hierarchical frequency sampling and periodicity-aware weighting.
- **[[source-frets|FreTS]]** — FreTS also operates in the frequency domain but uses MLPs rather than attention, achieving even greater efficiency at the cost of less sophisticated frequency selection.
- **[[timesnet]]** — TimesNet takes an alternative approach by transforming 1D series into 2D tensors for multi-periodicity modeling, a different paradigm from FEDformer's frequency-domain attention.

[^src-fedformer]: [[source-fedformer]]
