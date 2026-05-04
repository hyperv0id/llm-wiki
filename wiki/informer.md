---
title: "Informer"
type: entity
tags:
  - time-series
  - forecasting
  - transformer
  - efficient-attention
  - long-sequence
  - AAAI-2021
created: 2026-05-04
last_updated: 2026-05-04
source_count: 1
confidence: medium
status: active
---

# Informer

**Informer** is a pioneering efficient Transformer architecture for **Long Sequence Time-Series Forecasting (LSTF)** proposed by Haoyi Zhou et al., published at AAAI 2021 and recognized as the **Best Paper**. It is the first work to simultaneously address all three bottlenecks of applying vanilla Transformers to long sequences: quadratic time complexity, high memory usage, and slow autoregressive decoding speed[^src-zhou-informer-2021].

## Overview

Prior to Informer, Transformer-based models faced severe limitations for long-horizon forecasting. The canonical dot-product self-attention has $O(L^2)$ complexity, making it impractical for sequences longer than a few hundred steps. Stacking multiple layers compounds memory usage to $O(J \cdot L^2)$. Additionally, the standard encoder-decoder paradigm uses dynamic decoding (predicting one step at a time), which propagates errors and slows inference to RNN-like speeds[^src-zhou-informer-2021].

Informer overcomes all three issues with three complementary innovations: **ProbSparse self-attention** reduces complexity to $O(L \log L)$, **self-attention distilling** prunes redundant features to focus attention, and a **generative style decoder** predicts the entire sequence in one forward pass[^src-zhou-informer-2021].

## Key Innovations

### 1. ProbSparse Self-Attention

Informer's core insight is that self-attention score distributions exhibit a **long-tail pattern** — only a small fraction of query-key pairs contribute meaningfully to the output; the rest are near-uniform noise. This sparsity is quantified by measuring the KL divergence between each query's attention distribution and the uniform distribution. Informer selects only the **Top-$u$ queries** where $u = c \cdot \ln L_Q$, and for each selected query, computes attention over only the dominant keys. The remaining queries receive a mean value vector, effectively bypassing the quadratic bottleneck. Time and memory complexity drop to $O(L \log L)$ while preserving prediction quality[^src-zhou-informer-2021].

This differs from earlier sparse attention methods (e.g., Sparse Transformer's fixed stride patterns, LogSparse's exponential growing intervals) in that the sparsity is **data-driven**: which queries are "active" depends on the actual attention pattern of the input, not a hard-coded structure.

### 2. Self-Attention Distilling

To handle extremely long input sequences (e.g., 720+ time steps), Informer's encoder applies a **distilling operation** between ProbSparse attention layers: a 1D convolution with ELU activation followed by max-pooling with stride 2. This halves the temporal dimension at each layer, creating a **pyramid-like focus** where lower layers capture fine-grained patterns and higher layers distill the most salient features. The total encoder memory complexity is reduced to $O((2 - \epsilon) L \log L)$[^src-zhou-informer-2021].

### 3. Generative Style Decoder

The decoder receives a concatenated input: a **start token** (a known slice from the end of the encoder input) followed by **zero-padded placeholders** for the target prediction length. Masked ProbSparse attention prevents the decoder from attending to future positions (no auto-regressive leakage). The model then predicts the **entire output sequence in a single forward pass**, eliminating both the accumulation of prediction errors and the inference-time bottleneck of step-by-step decoding[^src-zhou-informer-2021].

## Architecture

Informer follows a stacked encoder-decoder structure:

```
Encoder:  [Input] → ProbSparse Attention → Distilling → ... → [Encoded Features]
Decoder:  [Start Token | Placeholders] → Masked ProbSparse Attention → [Output]
```

The distillation stacking in the encoder creates a multi-scale representation without additional parameters. The generative decoder is conceptually simple but practically powerful, offering orders-of-magnitude inference speedup over dynamic decoding.

## Performance

Compared against ARIMA, Prophet, LSTMa, LSTnet, DeepAR, LogTrans, and Reformer on four datasets (ETT, ECL, Weather), Informer achieved **state-of-the-art results** in both univariate and multivariate long-horizon forecasting. Ablation studies confirmed that ProbSparse, distilling, and the generative decoder each independently and jointly contributed to the performance gains[^src-zhou-informer-2021].

## Historical Significance

As the AAAI 2021 Best Paper, Informer established the **efficient Transformer for LSTF** research paradigm that subsequent works built upon. It serves as the primary baseline against which later architectures are measured:

- **[[autoformer|Autoformer]]** (NeurIPS 2021) — replaces self-attention entirely with Auto-Correlation, embeds decomposition as an inner module; demonstrates that its decomposition architecture generalizes to improve Informer's performance.
- **[[fedformer|FEDformer]]** (ICML 2022) — moves attention to the frequency domain via Fourier/Wavelet transforms, achieving $O(L)$ complexity.
- **[[timesnet|TimesNet]]** (ICLR 2023) — transforms 1D time series into 2D tensors based on multi-periodicity; lists Informer as a key baseline.
- **[[source-frets|FreTS]]** (NeurIPS 2023) — applies MLPs in the frequency domain; outperforms Informer by >20% MAE/RMSE.
- **[[dualsformer|Dualformer]]** (2026) — time-frequency dual domain learning; implicitly improves upon Informer's time-domain attention paradigm.
- **[[source-language-in-the-flow-of-time|TaTS]]** (ICLR 2026) — uses Informer as one of nine backbone models for plug-and-play text-augmented forecasting.

## Related Technical Concepts

- [[probsparse-self-attention]] — detailed technique description
- [[generative-style-decoder]] — one-forward-pass decoding technique
- [[lstf]] — Long Sequence Time-Series Forecasting concept

[^src-zhou-informer-2021]: [[source-zhou-informer-2021]]