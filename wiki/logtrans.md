---
title: "LogTrans"
type: entity
tags:
  - time-series
  - forecasting
  - transformer
  - sparse-attention
  - NeurIPS-2019
created: 2026-05-30
last_updated: 2026-05-30
source_count: 2
confidence: high
status: active
---

# LogTrans

**LogTrans** (LogSparse Transformer) is the first model to successfully apply the Transformer architecture to time series forecasting, published at NeurIPS 2019 by Li et al. (UCSB). It introduces two innovations that remain foundational to efficient Transformer-based forecasting: **convolutional self-attention** for local context awareness and **LogSparse self-attention** for breaking the quadratic memory bottleneck[^src-logtrans].

## Core Design

### Problem

Canonical Transformers face two barriers in time series forecasting[^src-logtrans]:

1. **Locality-agnostic**: Dot-product attention matches queries to keys based on pointwise values alone, missing local shape/context. A point could be an anomaly, a change point, or part of a repeating pattern — but the attention mechanism cannot distinguish these.
2. **Memory bottleneck**: $O(L^2)$ space complexity prevents modeling long sequences with fine granularity directly.

### Solution: Two Complementary Mechanisms

**Convolutional self-attention** replaces the linear projection (1×1 convolution) that produces queries and keys with a causal convolution of kernel size $k > 1$. This makes each query/key vector aware of its $k$-step local context before attention matching — so similarity is computed based on local shapes rather than isolated points. When $k=1$, it reduces to canonical self-attention[^src-logtrans].

[[logsparse-self-attention|LogSparse self-attention]] restricts each cell's attention to exponentially spaced predecessors: for cell $l$, the attendable set is $\{l - 2^{\lfloor \log_2 l \rfloor}, l - 2^{\lfloor \log_2 l \rfloor - 1}, \ldots, l-1, l\}$. Theorem 1 proves that stacking $\lfloor \log_2 L \rfloor + 1$ layers guarantees information flow between any pair of cells, while reducing per-layer memory from $O(L^2)$ to $O(L \log L)$ and total memory to $O(L (\log L)^2)$. The number of unique paths between distant cells grows super-exponentially in $\log_2(l - j)$, ensuring rich information flow for modeling delicate long-term dependencies[^src-logtrans].

Two optional refinements — local attention (dense attend to a $O(\log L)$-sized left window) and restart attention (dividing the sequence into subsequences) — create more paths without changing asymptotic complexity[^src-logtrans].

## Architecture

LogTrans uses a **decoder-only** Transformer architecture with learnable positional embeddings and time-series-ID embeddings. Covariates (year, month, day-of-week, hour, minute, age) are concatenated with these embeddings as input. A final fully-connected layer predicts Gaussian distribution parameters for the next time point[^src-logtrans].

```
Input: [z_{t-1} | x_t] ∈ R^{d+1}
  → Embedding (position + series-ID) + Covariate concat
  → N× Transformer Decoder layers (convolutional self-attn + LogSparse mask)
  → FC → (μ, σ) for next time point
```

All time series share the same model weights. Training maximizes log-likelihood over the full sequence[^src-logtrans].

## Key Results

- On **synthetic data**: Transformer sustains accuracy as dependency length grows ($t_0 = 24 \to 192$), while LSTM (DeepAR) deteriorates sharply after $t_0 \geq 96$. This validates Transformer's long-range modeling for forecasting[^src-logtrans].
- **Convolutional self-attention**: $k=9$ reduces $R_{0.5}$ loss by ~9% over $k=1$ on traffic-c. Training converges **faster and to lower losses** with larger kernels[^src-logtrans].
- **Sparse vs. full attention with equal memory**: On traffic-f, sparse + conv ($R_{0.5}=0.138$) substantially outperforms full attention ($0.161$) under the same memory budget, benefiting from the dataset's strong long-term dependencies[^src-logtrans].
- **Cross-dataset evaluation**: LogTrans achieves best overall performance on electricity-f, traffic-f, solar, wind (30-day horizon), and M4-Hourly against ARIMA, ETS, TRMF, DeepAR, and DeepState[^src-logtrans].

## Attention Pattern Analysis

Visualization of learned attention on traffic-c reveals automatic seasonality discovery[^src-logtrans]:
- **Layer 2**: Learns shared daily patterns
- **Layer 6**: Focuses on weekend-specific patterns
- **Layer 10**: Squeezes attention to only a few weekend cells — exhibiting extreme sparsity that motivates LogSparse design

In the final layer's attention matrix, weekday timestamps heavily attend to prior weekday timestamps at the same hour, while weekend timestamps cluster on prior weekend timestamps. The model autonomously discovers both hourly and daily seasonality without explicit architectural induction[^src-logtrans].

## Position in LTSF Lineage

LogTrans (NeurIPS 2019) is the pioneering work that opened the Transformer-for-time-series research line:

| Model | Year | Complexity | Key Innovation |
|-------|------|-----------|----------------|
| **LogTrans** | 2019 | $O(L (\log L)^2)$ | Convolutional self-attn + LogSparse (fixed exponential) |
| [[informer|Informer]] | 2021 | $O(L \log L)$ | ProbSparse (data-driven) + generative decoder |
| [[autoformer|Autoformer]] | 2021 | $O(L \log L)$ | Decomposition + Auto-Correlation |
| [[fedformer|FEDformer]] | 2022 | $O(L)$ | Frequency-enhanced attention |
| [[patchtst|PatchTST]] | 2023 | $O((L/S)^2)$ | Patch tokenization + CI |
| [[itransformer|iTransformer]] | 2024 | $O(N^2)$ | Inverted dimensions |

The Informer paper explicitly cites LogTrans as both a predecessor and baseline. The LogSparse fixed exponential pattern is superseded by [[probsparse-self-attention|ProbSparse]]'s data-driven query selection, but the insight that attention exhibits exploitable sparsity in time series remains central[^src-zhou-informer-2021].

## Limitations

- Autoregressive decoding is slower than [[informer|Informer]]'s one-forward-pass [[generative-style-decoder]]
- The fixed LogSparse pattern treats all sequences identically — no adaptation to varying dependency structures
- No explicit modeling of cross-variate interactions (each variate is independently processed by the shared decoder)
- Gaussian likelihood assumption limits distributional flexibility

[^src-logtrans]: [[source-logtrans]]
[^src-zhou-informer-2021]: [[source-zhou-informer-2021]]
