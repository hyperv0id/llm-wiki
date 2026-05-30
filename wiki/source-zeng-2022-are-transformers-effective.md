---
title: "Are Transformers Effective for Time Series Forecasting?"
type: source-summary
tags:
  - time-series
  - forecasting
  - transformer-critique
  - linear-model
  - LTSF
created: 2026-05-30
last_updated: 2026-05-30
source_count: 1
confidence: high
status: active
---

# Are Transformers Effective for Time Series Forecasting?

**Authors**: Ailing Zeng, Muxi Chen (CUHK), Lei Zhang (IDEA), Qiang Xu (CUHK). arXiv:2205.13504v3, August 2022.

## Core Thesis

This paper **challenges the validity** of Transformer-based solutions for long-term time series forecasting (LTSF). The central argument: self-attention is permutation-invariant and therefore inherently loses temporal ordering information — the very information that matters most in time series. While positional encoding and sub-series tokenization preserve *some* ordering, the nature of self-attention inevitably results in temporal information loss[^src-zeng-2022-are-transformers-effective].

## Key Contributions

1. **First critique of LTSF-Transformers**: Questions the booming Transformer-based LTSF research line, arguing temporal modeling capabilities are exaggerated on existing benchmarks.
2. **LTSF-Linear baseline**: Introduces embarrassingly simple one-layer linear models (DLinear, NLinear, Vanilla Linear) that outperform all existing Transformer-based LTSF models on nine benchmarks, often by 20%–50%[^src-zeng-2022-are-transformers-effective].
3. **Comprehensive empirical studies**: Examines long-input capability, order sensitivity, embedding strategy impacts, and efficiency of existing LTSF-Transformers.

## LTSF-Linear Models

Three variants of single-layer linear models:
- **Vanilla Linear**: $\hat{X}_i = W X_i$ where $W \in \mathbb{R}^{T \times L}$. Shares weights across variates; no spatial correlation modeling.
- **DLinear**: Decomposes input into trend (moving average) and seasonal (remainder) components, applies separate linear layers, and sums results. Explicitly handles trend.
- **NLinear**: Subtracts last value of input sequence before linear layer, then adds it back. Simple normalization handling distribution shift.

## Critical Findings

### Transformers Fail to Exploit Long Input
Existing Transformer-based LTSF models' performance **deteriorates or stays stable** when the look-back window size increases from 96 to 720. In contrast, LTSF-Linear's performance **significantly improves** with larger look-back windows, indicating Transformers overfit temporal noises rather than extracting useful temporal information[^src-zeng-2022-are-transformers-effective].

### Self-Attention Is Not Effective for LTSF
Gradually simplifying Informer to a linear model (Informer → Att.-Linear → Embed+Linear → Linear) **improves** performance on Exchange-Rate (MSE: 0.847 → 0.084 at T=96), demonstrating self-attention and other complex modules are unnecessary for existing LTSF benchmarks[^src-zeng-2022-are-transformers-effective].

### Temporal Order Not Well Preserved
Shuffling input sequences randomly barely affects Transformer performance on Exchange-Rate (average drop: -0.09% to 1.98%), while LTSF-Linear drops 27.26%–46.81%. This indicates Transformers with positional/temporal embeddings preserve quite limited temporal relations and are prone to overfit on noisy data[^src-zeng-2022-are-transformers-effective].

### DMS vs. IMS Forecasting
Prior Transformer papers compared against IMS (autoregressive) baselines which suffer error accumulation. The DMS strategy alone accounts for much of the apparent improvement. Even naive Repeat (last value) outperforms all Transformers on Exchange-Rate by ~45%[^src-zeng-2022-are-transformers-effective].

### Training Data Size Not the Bottleneck
Reducing Traffic training data from full (17,544×0.7 hours) to one year (8,760 hours) actually *lowers* errors for FEDformer and Autoformer in most cases, suggesting model capacity, not data, is the limiting factor[^src-zeng-2022-are-transformers-effective].

### Efficiency Claims Questioned
Most Transformer variants achieve similar or worse practical inference time vs. vanilla Transformer. DLinear: 0.04G MACs, 139.7K params, 0.4ms inference; Informer: 3.93G MACs, 14.39M params, 49.3ms[^src-zeng-2022-are-transformers-effective].

## FEDformer's Competitive Case

FEDformer achieves competitive results on ETTh1 because it employs classical time series analysis techniques (frequency processing) that introduce **time series inductive bias**, benefiting temporal feature extraction. This finding suggests the path forward is incorporating domain-specific inductive bias rather than relying on generic self-attention[^src-zeng-2022-are-transformers-effective].

## Limitations

LTSF-Linear has limited model capacity — one linear layer cannot capture temporal dynamics from change points. The paper advocates for new model designs, data processing, and benchmarks.

## Connections

- Directly critiques [[informer]], [[autoformer]], [[fedformer]], Pyraformer, LogTrans
- Reinforces the importance of [[lstf|LSTF]] problem formulation
- DLinear's decomposition scheme follows [[autoformer|Autoformer]]'s moving average approach
- FEDformer's competitive case supports the value of [[frequency-enhanced-block|frequency-domain inductive bias]]
- NLinear's normalization technique anticipates later [[instance-normalization|RevIN-style]] approaches

[^src-zeng-2022-are-transformers-effective]: [[source-zeng-2022-are-transformers-effective]]
