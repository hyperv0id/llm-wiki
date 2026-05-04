---
title: "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"
type: source-summary
tags:
  - time-series
  - forecasting
  - transformer
  - efficient-attention
  - long-sequence
  - AAAI-2021
created: 2026-05-04
last_updated: 2026-05-04
source_count: 0
confidence: medium
status: active
---

# Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting

## Overview

Published at **AAAI 2021** (Best Paper) by Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang, Informer addresses the problem of **Long Sequence Time-Series Forecasting (LSTF)** — predicting long-horizon future values from long historical sequences. The paper identifies three critical bottlenecks of applying vanilla Transformers to LSTF: (1) quadratic time complexity $O(L^2)$ of canonical self-attention; (2) $O(J \cdot L^2)$ memory usage from stacking $J$ layers; and (3) slow inference speed from the step-by-step dynamic decoding paradigm. Informer introduces three distinct innovations to overcome all three bottlenecks simultaneously.

## Key Method

### 1. ProbSparse Self-Attention

The key insight is that self-attention score distributions follow a **long-tail pattern**: only a few dot-product pairs contribute to major attention; the rest are negligible. Informer formalizes this via a sparsity measurement $M(q_i, K) = \max_j\{q_i k_j^\top / \sqrt{d}\} - \frac{1}{L_K}\sum_j q_i k_j^\top / \sqrt{d}$ derived from KL divergence between the query's attention distribution and the uniform distribution. Only the **Top-$u$ queries** where $u = c \cdot \ln L_Q$ need to compute full dot-product attention; the remaining queries attend to only the $U = u \cdot \ln L_K$ dominant keys. This reduces time and memory complexity to **$O(L \log L)$** without sacrificing performance.

### 2. Self-Attention Distilling (Encoder)

To handle extremely long input sequences, the encoder applies a **distilling operation** after each ProbSparse attention block: 1D convolution (ELU activation) followed by max-pooling with stride 2, which halves the time dimension of the input at each layer. The distilling process highlights dominant attention features and creates a focused self-attention feature map in higher layers, reducing total encoder space complexity to $O((2 - \epsilon) L \log L)$.

### 3. Generative Style Decoder

Instead of the conventional step-by-step (dynamic) decoding that propagates cumulative errors and limits speed, Informer adopts a **generative inference paradigm**: the decoder receives a start token (a slice of the input sequence concatenated with placeholders for target positions) and predicts the **entire output sequence in one forward pass**. Masked ProbSparse attention in the decoder prevents auto-regressive information leakage, enabling extremely fast long-sequence prediction.

### Architecture

Informer follows a stacked encoder-decoder structure. The encoder processes the full long-sequence input through alternating ProbSparse attention + distilling blocks. The decoder takes a long sequence input (start token + zero-padded target) and maps masked outputs to the final prediction via a fully connected layer. The model is trained with MSE loss.

## Results

Informer was evaluated on four datasets:
- **ETT** (Electricity Transformer Temperature): 2-year data at two granularities (ETTh1, ETTh2 hourly; ETTm1 15-min)
- **ECL** (Electricity Consuming Load): 2-year hourly data from 321 clients
- **Weather**: 4-year climatological data for ~1,600 U.S. locations

Comprehensive experiments compared against ARIMA, Prophet, LSTMa, LSTnet, DeepAR, LogTrans, and Reformer. Informer **significantly outperformed all baselines** in both univariate and multivariate settings, demonstrating clear superiority in long-range dependency capture and inference speed. Ablation studies confirmed that each component (ProbSparse, distilling, generative decoder) independently and jointly contributes to the performance gains.

## Critique

- **Strengths**: Informer was the first to systematically address all three Transformer bottlenecks for LSTF simultaneously — computational complexity, memory usage, and inference speed. The probabilistic sparsity insight (long-tail attention distribution) is theoretically elegant and practically effective. AAAI 2021 Best Paper recognition underscores its impact. The generative decoder concept anticipates the trend toward non-autoregressive sequence generation.
- **Limitations**: ProbSparse relies on sampling-based approximate query selection, which may miss some informative queries in edge cases. The distilling operation reduces temporal resolution, potentially losing fine-grained patterns that shorter-horizon models could capture. The ETT/ECL/Weather benchmarks, while standard, do not include modern large-scale datasets (e.g., Traffic, ILI). Compared to later models like [[autoformer|Autoformer]] (which integrates decomposition as an inner module and replaces self-attention entirely with Auto-Correlation) and [[fedformer|FEDformer]] (which uses frequency-domain attention), Informer still uses a dot-product-based attention variant rather than a fundamentally different aggregation mechanism.
- **Historical significance**: Informer is a seminal work that paved the way for the Transformer-based LSTF research line, directly influencing Autoformer, FEDformer, TimesNet, and many subsequent models. It serves as the primary efficient-Transformer baseline against which later architectures are measured.

## Related Pages

- [[informer]] — entity page
- [[probsparse-self-attention]] — core attention mechanism
- [[generative-style-decoder]] — one-forward-pass decoding technique
- [[lstf]] — Long Sequence Time-Series Forecasting concept
- [[autoformer]] — builds on Informer's Transformer foundation with decomposition
- [[fedformer]] — extends with frequency-domain attention