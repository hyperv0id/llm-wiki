---
title: "LogTrans: Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting"
type: source-summary
tags:
  - time-series
  - forecasting
  - transformer
  - sparse-attention
  - convolutional-attention
  - NeurIPS-2019
created: 2026-05-30
last_updated: 2026-05-30
source_count: 1
confidence: high
status: active
---

# LogTrans: Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting

- **Authors**: Shiyang Li, Xiaoyong Jin, Wenhu Chen, Yao Xuan, Yu-Xiang Wang, Xiyou Zhou, Xifeng Yan (University of California, Santa Barbara)
- **Venue**: NeurIPS 2019
- **arXiv**: 1907.00235v3

## Core Contribution

LogTrans is the first paper to successfully apply the Transformer architecture to time series forecasting. It identifies and addresses two fundamental weaknesses of the canonical Transformer for this domain: (1) **locality-agnosticism** — pointwise dot-product attention is insensitive to local context such as shape, making it prone to anomalies; and (2) **memory bottleneck** — quadratic $O(L^2)$ space complexity makes modeling long, fine-grained time series infeasible[^src-logtrans].

Two innovations solve these problems:

1. **Convolutional self-attention**: Replaces the standard linear projections for queries and keys with causal convolutions of kernel size $k$ (stride 1). This makes query-key matching aware of local context (shapes, not just pointwise values), leading to lower training loss and better forecasting accuracy. When $k=1$, it degrades to canonical self-attention — thus it is a strict generalization[^src-logtrans].

2. **LogSparse self-attention**: Each cell attends only to cells at exponentially growing intervals and itself: $I_l^k = \{l - 2^{\lfloor \log_2 l \rfloor}, l - 2^{\lfloor \log_2 l \rfloor - 1}, \ldots, l - 2^0, l\}$. With $\lfloor \log_2 L \rfloor + 1$ stacked layers, information can still flow between any pair of cells (Theorem 1), while per-layer memory drops to $O(L \log L)$ and total memory to $O(L (\log L)^2)$. Two optional extensions — local attention (dense window of size $O(\log L)$) and restart attention (dividing the sequence into subsequences) — create more information paths without changing complexity[^src-logtrans].

## Key Results

On synthetic data (piecewise sinusoidal signals with varying $t_0$): canonical Transformer maintains accuracy as $t_0$ grows, while DeepAR (LSTM-based) drops sharply when $t_0 \geq 96$, demonstrating Transformer's superior long-term dependency capture[^src-logtrans].

On real-world datasets (electricity, traffic at fine/coarse granularities, solar, wind, M4-Hourly):

- **Convolutional self-attention**: Up to 9% relative $R_{0.5}$ improvement on traffic-c with $k=9$ vs. $k=1$. Training curves converge faster and to lower loss with larger kernels[^src-logtrans].
- **Memory-constrained comparison**: LogSparse models (input length 768 on electricity-f) match or outperform full-attention counterparts (input length 293) despite comparable memory. On traffic-f, sparse + conv achieves $R_{0.5}=0.138$ vs. full attention's $0.161$[^src-logtrans].
- **Same input length**: Full attention generally outperforms sparse as expected, but on traffic-f with strong long-term dependencies, sparse + conv ($R_{0.5}=0.138$) slightly edges full attention ($0.147$), indicating LogSparse captures long-range dependencies effectively even with fewer computed dot-products[^src-logtrans].
- LogTrans achieves best overall results across all datasets compared to ARIMA, ETS, TRMF, DeepAR, and DeepState baselines[^src-logtrans].

## Architecture Details

The model uses a **Transformer decoder-only** architecture (no encoder), with learnable positional embeddings and time-series-ID embeddings summed before concatenation with covariates (year, month, day-of-week, hour, minute, age). A final fully-connected layer outputs Gaussian likelihood parameters with appropriate activation transformations for the next time point. The training objective maximizes log-likelihood over the entire sequence, including both history and forecast horizon[^src-logtrans].

Attention pattern analysis on traffic-c reveals that Transformer automatically learns both hourly and daily seasonality: weekday points attend heavily to previous weekdays at the same hour, while weekend points focus on previous weekends[^src-logtrans].

## Limitations

- Decoder-only autoregressive design means training and inference are slower than one-forward-pass generative decoders like [[informer|Informer]]'s.
- The LogSparse pattern is structurally fixed (exponential step sizes), unlike data-driven sparsity in [[probsparse-self-attention]].
- Does not address channel interaction across multivariate time series — each series is modeled independently through the shared decoder.
- Evaluated only on univariate-time-series-per-channel forecasting; cross-variate dependencies are not explicitly modeled (cf. [[itransformer]], [[crossformer]]).
- Gaussian likelihood assumption may not fit all data distributions; the paper acknowledges negative-binomial as an alternative for count data.

## Historical Significance

LogTrans (NeurIPS 2019) precedes [[informer|Informer]] (AAAI 2021) and establishes the Transformer-for-time-series paradigm. It is the direct predecessor of Informer, which cites LogTrans as a primary baseline and improves upon its fixed LogSparse pattern with data-driven [[probsparse-self-attention]]. The convolutional self-attention idea — using convolutions to inject local context into attention — also foreshadows later work on locality-aware attention mechanisms in time series.

Subsequent works, notably the [[ltsf-linear]] critique (Zeng et al., 2022), challenged the Transformer paradigm that LogTrans helped initiate. [[patchtst|PatchTST]] later showed that patching + channel independence can make Transformers competitive again.

[^src-logtrans]: [[source-logtrans]]
