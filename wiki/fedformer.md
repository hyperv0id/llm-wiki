---
title: "FEDformer"
type: entity
tags:
  - time-series
  - forecasting
  - frequency-domain
  - transformer
  - fourier
  - wavelet
  - ICML-2022
  - alibaba
created: 2026-04-28
last_updated: 2026-05-04
source_count: 2
confidence: high
status: active
---

# FEDformer

**FEDformer** (Frequency Enhanced Decomposed Transformer) is a Transformer architecture for [[lstf|long-term time series forecasting]], proposed by Tian Zhou, Ziqing Ma, **Qingsong Wen**, Xue Wang, Liang Sun, and Rong Jin from **Alibaba DAMO Academy**, published at **ICML 2022**[^src-fedformer]. It was the first Transformer-based forecasting model to achieve $O(L)$ linear complexity while simultaneously improving accuracy over prior SOTA.

## Core Insight

Standard Transformers make point-wise independent predictions, failing to preserve the **global distribution** of time series (overall trend). FEDformer addresses this with two complementary mechanisms[^src-fedformer]:

1. **Frequency-domain attention** — operates on Fourier/Wavelet transforms rather than raw time-domain signals, capturing global spectral properties
2. **Learnable seasonal-trend decomposition (MOEDecomp)** — uses a mixture-of-experts of average filters to adaptively extract trend, ensuring output distribution matches input

A key theoretical contribution: **randomly selecting a fixed number of Fourier modes** (including both high and low frequencies) is provably better than keeping only low-frequency components. Theorem 1 bounds the reconstruction error: $|A - P_{A'}(A)| \leq (1+\epsilon)|A - A_k|$ when sampling $s = O(k^2/\epsilon^2)$ modes randomly, leveraging the low-rank property of multivariate time series in the frequency domain[^src-fedformer].

## Architecture

FEDformer follows an encoder-decoder structure with three novel components replacing standard Transformer blocks[^src-fedformer]:

### 1. [[frequency-enhanced-block|Frequency Enhanced Block (FEB)]]
Replaces self-attention in both encoder and decoder. Two variants[^src-fedformer]:
- **FEB-f** (Fourier): DFT → random M-mode selection → learnable kernel $R$ → inverse DFT, $O(L)$ complexity
- **FEB-w** (Wavelet): recursive multiwavelet decomposition (Legendre basis) with 3 shared FEB-f modules, $L=3$ cycles

### 2. [[frequency-enhanced-attention|Frequency Enhanced Attention (FEA)]]
Replaces cross-attention between encoder and decoder[^src-fedformer]:
- **FEA-f**: Transforms $q, k, v$ via DFT separately, attention in frequency domain: $\sigma(\tilde{Q} \cdot \tilde{K}^\top) \cdot \tilde{V}$, then inverse DFT
- **FEA-w**: Same recursive wavelet decomposition applied to $q, k, v$ independently, using FEA-f modules

### 3. [[moe-decomposition|Mixture of Experts Decomposition (MOEDecomp)]]
Data-adaptive trend extraction using multiple average filters of different sizes + learnable softmax mixing weights. Improves over single fixed-window decomposition by **2.96% on average**[^src-fedformer].

### Encoder-Decoder Flow
$$
\begin{aligned}
\text{Encoder}^l &: \text{FEB} \to \text{MOEDecomp} \to \text{FFN} \to \text{MOEDecomp} \\
\text{Decoder}^l &: \text{FEB} \to \text{MOEDecomp} \to \underbrace{\text{FEA}(\text{enc outputs})}_{\text{cross-attention}} \to \text{MOEDecomp} \to \text{FFN} \to \text{MOEDecomp}
\end{aligned}
$$

Final output: $\text{prediction} = W_S \cdot X_{de}^M + T_{de}^M$, where trend $T_{de}^l = T_{de}^{l-1} + \sum_i W_{l,i} \cdot T_{de}^{l,i}$[^src-fedformer].

## Complexity

FEDformer achieves $O(L)$ linear time and memory complexity—the first Transformer-based LSTF model to do so—by[^src-fedformer]:
- Pre-selecting $M=64$ modes before DFT (avoids $O(L \log L)$ FFT)
- Fixed recursive depth $L=3$ for wavelet variant
- Frequency-domain operation **decouples** sequence length from attention matrix dimension

| Model | Time | Memory | Test Steps |
|:------|:-----|:-------|:-----------|
| **FEDformer** | **$O(L)$** | **$O(L)$** | **1** |
| Autoformer | $O(L \log L)$ | $O(L \log L)$ | 1 |
| Informer | $O(L \log L)$ | $O(L \log L)$ | 1 |
| Transformer | $O(L^2)$ | $O(L^2)$ | $L$ |

## Performance

Evaluated on 6 benchmarks with input $I=96$, prediction $O \in \{96,192,336,720\}$[^src-fedformer]:

| Metric | vs. Autoformer | Notes |
|:-------|:---------------|:------|
| Multivariate MSE | **-14.8%** | Best on all 6 datasets, all horizons |
| Univariate MSE | **-22.6%** | Traffic/Weather over -30% |
| Low-periodicity (Exchange/ILI) | **>-20%** | Works without clear periodicity |
| KS distribution test | **All P > 0.01** | Only model retaining input distribution |
| Ablation (FEB+FEA) | **16/16 cases** | Full model beats all stripped variants |

Random mode selection consistently outperforms fixed low-frequency selection across all $M \in \{2, \dots, 256\}$[^src-fedformer].

## Connections

### Precursors
- **[[informer]]** (AAAI 2021 Best Paper) — pioneered efficient Transformers for LSTF ($O(L \log L)$ via ProbSparse attention + distilling); FEDformer further reduces to $O(L)$[^src-informer]
- **[[autoformer]]** (NeurIPS 2021) — introduced progressive seasonal-trend decomposition inside Transformer + Auto-Correlation mechanism; FEDformer inherits the decomposition architecture and extends it with learnable MOEDecomp[^src-fedformer]

### Successors & Alternatives
- **[[dualsformer|Dualformer]]** — builds on FEDformer's frequency-domain paradigm but replaces fixed random sampling with input-adaptive hierarchical frequency selection and periodicity-aware weighting, addressing FEDformer's main limitation[^src-fedformer]
- **[[source-frets|FreTS]]** (NeurIPS 2023) — takes frequency-domain operation further by using pure MLPs instead of attention, achieving greater efficiency at the cost of coarser frequency selection
- **[[hyperd|HyperD]]** — uses explicit short/long-term periodicity decoupling with separate encoder pathways rather than uniform frequency processing
- **[[timesnet]]** (ICLR 2023) — alternative paradigm: transforms 1D series into 2D tensors for multi-periodicity modeling via FFT-based period discovery
- **[[sparsetsf|SparseTSF]]** (ICML 2024 Oral) — pushes efficiency to extreme: sub-1k parameters via cross-period sparse forecasting
- **[[cyclenet|CycleNet]]** (NeurIPS 2024) — explicit periodicity modeling via learnable recurrent cycles, a different decomposition philosophy

### Broader Impact
- **[[tslib]]** — benchmarks FEDformer as a representative frequency-domain model
- **[[periodicity-modeling-in-time-series|周期建模文献]]** — positions FEDformer in the evolution from efficiency-first (Informer) to structure-first (Autoformer) to domain-specific (FEDformer) approaches
- **[[traffic-forecasting|交通预测]]** — FEDformer applied to traffic but treats frequencies uniformly; HyperD and others later improved upon this
- **[[lstf|LSTF]]** — FEDformer represents the domain-specific stage in LSTF evolution

[^src-fedformer]: [[source-fedformer]]
[^src-informer]: [[source-zhou-informer-2021]]