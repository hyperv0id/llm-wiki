---
title: "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting"
type: source-summary
tags:
  - time-series
  - forecasting
  - frequency-domain
  - transformer
  - fourier-transform
  - wavelet-transform
  - seasonal-trend-decomposition
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# FEDformer

**FEDformer** (Frequency Enhanced Decomposed Transformer) is a Transformer architecture for long-term series forecasting, proposed by Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin (Alibaba Group), published at ICML 2022. It combines seasonal-trend decomposition with frequency-domain attention mechanisms to capture both global time series properties and detailed structures, while achieving linear complexity.

## Overview

Standard Transformer-based forecasting methods struggle to maintain the global distributional properties of time series (e.g., overall trend) because they make point-wise predictions independently per timestep[^src-fedformer]. FEDformer addresses this by integrating two key ideas: (1) seasonal-trend decomposition via a Mixture-of-Experts (MOE) scheme to separately model global trends and seasonal patterns, and (2) frequency-enhanced attention blocks that operate in the Fourier or Wavelet domain rather than the time domain. The paper provides both theoretical and empirical evidence that randomly selecting a fixed number of frequency components (including both high and low frequencies) yields better representations than keeping only low frequencies—the conventional wisdom.

## Key Method

FEDformer's architecture is an encoder-decoder structure with three novel components[^src-fedformer]:

1. **Frequency Enhanced Block (FEB)**: Substitutes the self-attention block. Two variants exist—FEB-f uses Discrete Fourier Transform (DFT) to project inputs into the frequency domain, randomly selects M modes, applies a parameterized kernel, and returns to the time domain via inverse DFT. FEB-w uses multiwavelet decomposition with Legendre polynomial bases, processing signals recursively across scales.

2. **Frequency Enhanced Attention (FEA)**: Substitutes the cross-attention block between encoder and decoder. FEA-f converts queries, keys, and values via DFT, applies attention in the frequency domain (softmax/tanh on Q̃·K̃⊤, then multiplied by Ṽ), and inverts back. FEA-w applies the same recursive multiwavelet decomposition to q, k, v separately.

3. **Mixture of Experts Decomposition (MOEDecomp)**: Replaces fixed-window average pooling with a learnable set of average filters of different sizes, combined via data-dependent softmax weights. This adaptively extracts trend components from complex periodic patterns.

By keeping only M randomly selected Fourier modes (default M=64), the model achieves O(L) time and memory complexity—linear in sequence length—compared to O(L²) for standard Transformer.

## Results

FEDformer was evaluated on six benchmark datasets (ETTm2, Electricity, Exchange, Traffic, Weather, ILI) and compared against Autoformer, Informer, LogTrans, and Reformer[^src-fedformer]. It achieved the best performance across all datasets and all horizons. Multivariate MSE was reduced by 14.8% relative to Autoformer; univariate MSE reduction was 22.6%. On datasets without clear periodicity (Exchange, ILI), improvements exceeded 20%. The Kolmogorov-Smirnov test confirmed that FEDformer's outputs share a more similar distribution with inputs than competing models. Random mode selection outperformed fixed low-frequency selection across all mode counts (M ∈ {2, 4, ..., 256}).

## Critique

FEDformer was a seminal work that demonstrated the effectiveness of frequency-domain operations for time series Transformers, inspiring subsequent models like Dualformer and FreTS. Its key insight—random Fourier mode selection is both theoretically justified and practically superior to low-pass filtering—was well-supported. The wavelet variant provides complementary benefits for capturing local structures. However, the paper's baselines did not include some later strong models (PatchTST, iTransformer), and the ILI dataset results show relatively high absolute errors. The MOEDecomp block adds architectural complexity, and the choice between Fourier and wavelet variants requires dataset-specific tuning. The fixed random mode selection, while efficient, is not input-adaptive—a limitation later addressed by models like Dualformer's periodicity-aware weighting.

[^src-fedformer]: [[source-fedformer]]
