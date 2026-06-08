---
title: "T1 — One-to-One Channel-Head Binding for Multivariate Time-Series Imputation"
type: source-summary
tags:
  - time-series
  - data-imputation
  - cnn-transformer
  - channel-head-binding
  - iclr-2026
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

**T1** ("Time series imputation with 1-to-1 channel-head binding") is a CNN-Transformer hybrid for multivariate time series imputation by Dongik Park, Hyunwoo Ryu, Suahn Bae, Keondo Park, and Hyung-Sin Kim (Seoul National University), published at **ICLR 2026** (arXiv:2602.21043v4)[^src-t1]. Code: github.com/Oppenheimerdinger/T1.

## Problem

Robust imputation under heavy/diverse missingness requires *two* abilities simultaneously: (1) reconstructing temporal structure from sparse observations *within* each variable, and (2) selectively transferring complementary information *across* variables without importing noise. When temporal features are corrupted by missingness, naïve cross-variable transfer amplifies errors[^src-t1].

## Taxonomy of Prior Compromises

T1 frames existing architectures as each excelling at one ability while compromising the other[^src-t1]:
- **(i) Time-axis tokenization** (Vanilla Transformer, SAITS diagonal-masked): missing values directly corrupt per-timestep tokens, contaminating all computations.
- **(ii) Variable-axis tokenization** ([[itransformer|iTransformer]]): fuses all temporal patterns of a variable into one token, losing feature-level selectivity.
- **(iii) Dual-axis tokenization** ([[imputeformer|ImputeFormer]], [[csdi|CSDI]], Crossformer): attention on both axes struggles when missing values block intermediate pathways.
- **(iv) Temporal CNN** (TimesNet, ModernTCN): strong multi-scale temporal features but limited cross-variable transfer.

## Method

T1 is a **task-aligned hybrid**: CNNs extract temporal features within variables; attention performs selective cross-variable transfer[^src-t1]. Three components:

1. **Mask-Aware Embedding** — instance-normalize each series using *observed-only* mean/std, stack normalized series + observation mask into a 2-channel input, encode with a strided 1D conv (C filters) plus a learnable variable encoding.
2. **T1 Blocks (×B)** — (a) Temporal Convolutional Q/K/V Projection via depthwise convs whose weights are *shared across variables* (so each channel extracts the same pattern type → semantically-aligned channels), with parallel large+small kernels for multi-scale; (b) **Channel-Head Binding (CHead Attention)** — the key mechanism (see [[channel-head-binding]]); (c) Convolutional FFN (pointwise, inverted bottleneck).
3. **Reconstruction Upsampler** — parameter-free PixelShuffle1D (rearranges channels→time, avoids checkerboard artifacts) + pointwise conv, then denormalize.

Trained self-supervised: sequence length 96, 40% of observed values randomly masked as reconstruction targets, MSE loss. **Consistent hyperparameters across all datasets**[^src-t1].

## Channel-Head Binding (Core Contribution)

CHead Attention creates a **one-to-one correspondence between CNN channels and attention heads** (n_h = C): each head processes only its bound channel across all variables[^src-t1]. When missingness prevents a channel from observing its specialized pattern, the corresponding head tempers reliance on that channel, while feature-level isolation prevents localized uncertainty from contaminating other channels.

## Results

On 11 datasets (9 benchmark + PhysioNet2012, AQI36) vs 11 baselines[^src-t1]:
- **Point missing** (avg over 0.1/0.3/0.5/0.7): SOTA, **−46% MSE** vs second-best PatchTST, −56% vs specialized imputer PSW-I.
- **Extreme sparsity (0.7)**: T1 MSE 0.049 ≈ half of PatchTST 0.092; trained at 0.4, generalizes to unseen 0.1–0.7 without retraining.
- **Block missing** (5% point + 0.15% block of 24–96 steps): **−48% MSE** vs PatchTST.
- **Natural missing**: PhysioNet2012 (up to 94% total missing) −23% vs DLinear; AQI36 −13% vs PatchTST.
- **Efficient**: only 0.543M parameters (vs SAITS 5.27M, PatchTST 2.19M, CSDI 1.20M).

## Key Findings

- Removing cross-variable modeling entirely costs **+56.16% MSE**; replacing CHead attention with pointwise conv costs **+12.91%** — *both that cross-variable info matters and that adaptive (attention) transfer beats fixed (conv) transfer*[^src-t1].
- 1-to-1 binding is crucial: 8/16/32 channels-per-head degrade by +7.45%/+16.86%/+14.57%[^src-t1].
- Representation analysis: attention to a target variable drops as its missing ratio rises (shallow layers most: layer 1 −46%, last layer −6%); modulation depends on *which* patterns remain observable (removing high-variance regions −10.4% vs low-variance −7.5%), not just the missing ratio[^src-t1].

## Limitations / Future Work

Fixed sequence length 96. Future: online streaming real-time imputation; active sensing for sensor selection under resource constraints[^src-t1].

[^src-t1]: [[source-t1]]
