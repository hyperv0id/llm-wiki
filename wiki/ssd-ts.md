---
title: "SSD-TS"
type: entity
tags:
  - diffusion-models
  - state-space-model
  - mamba
  - time-series-imputation
  - probabilistic-modeling
  - kdd-2025
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# SSD-TS (State Space Diffusion for Time Series)

**SSD-TS** is a probabilistic time series imputation model that uses the [[mamba|Mamba]] selective state space model as the denoising backbone within a conditional [[diffusion-model|diffusion framework]], published at KDD 2025 (Hongfan Gao et al., ECNU) [^src-ssdts]. It is the first method to replace Transformer/S4 attention backbones with Mamba in a diffusion model for time series imputation.

## Core Architecture

SSD-TS follows the conditional DDPM framework ($\epsilon_\theta(x_t, t \mid X^{co})$), but replaces the self-attention backbone with Mamba-based modules [^src-ssdts]:

```
Observed Series X_co ──► Input SMM Blocks ──► Conditional SMM Blocks ──► Noise Prediction ε̂
```

Each **SMM (Sequential Mamba Module)** contains stacked pairs of:
- **[[bam|BAM]]** (Bidirectional Attention Mamba) — intra-channel dependency modeling
- **[[cmb|CMB]]** (Channel Mamba Block) — inter-channel dependency modeling

The basic building block, **PNM (Parallel Mamba Block)**, uses forward+backward convolution + SSM scans with gating, normalization, and residual connections, directly replacing Transformer self-attention [^src-ssdts].

## Why Mamba over Attention for Diffusion

SSD-TS identifies two key advantages of Mamba over self-attention in the diffusion context [^src-ssdts]:

1. **Content-independent parameter updates**: During early diffusion steps (data ≈ pure noise), attention weights derived from noisy input similarity are misleading. SSM's state update $h_t = A_t h_{t-1} + B_t x_t$ is independent of content similarity, and gating mechanisms help filter noise.

2. **Controllable frequency response**: SSMs have frequency response $\hat{K}(\omega) = C(i\omega I-A)^{-1}B$, enabling selective suppression of broadband noise while preserving signal in specific frequency bands. This is impossible with frequency-unbiased attention.

## Performance

On standard imputation benchmarks [^src-ssdts]:

| Dataset | Key Result |
|---------|-----------|
| MuJoCo (90% missing) | MSE $6.5\times10^{-4}$ — 65.8% improvement over SSSD |
| PhysioNet (10%/50%/90%) | Best RMSE across all rates (0.339/0.509/0.623) |
| AQI | RMSE 18.66, second only to D³M |
| ETTm1 Forecasting | Ranks 1st in 3/4 metrics |

Probabilistic calibration (CRPS): best in 3/4 tasks (10%/50% PhysioNet + AQI).

**Efficiency**: 87.57M parameters but 1.6× faster inference and 2.5× lower GPU memory than a Transformer-backbone variant (4536MB vs 11250MB) [^src-ssdts].

## Key Design Choices

- **Bidirectional** > unidirectional for intra-channel modeling
- **Temporal attention** is the most impactful component (largest ablation drop)
- **CMB with Mamba** > Channel Attention (SENet) for inter-channel dependency — proves SSMs are effective even for channel modeling
- **$C=128$ channels** balances performance and single-GPU memory capacity

## Compared to Related Methods

| Method | Backbone | Channel Modeling | Temporal Modeling |
|--------|----------|-----------------|-------------------|
| [[csdi|CSDI]] | Transformer | Feature Transformer (attention) | Time Transformer (attention) |
| SSSD | S4 SSM | S4 | S4 |
| D³M | Transformer + EMA | Attention + gating | Attention + gating |
| **SSD-TS** | **Mamba** | **CMB (unidirectional Mamba)** | **BAM (bidirectional Mamba + temporal attention)** |

## Limitations

- Only Mamba evaluated; Mamba-2 and other SSM variants not tested
- $C=128$ channel limit constrained by single-GPU memory
- Code: https://github.com/decisionintelligence/SSD-TS

## Related Pages

- [[bam|BAM (Bidirectional Attention Mamba)]]
- [[cmb|CMB (Channel Mamba Block)]]
- [[mamba|Mamba]] — the backbone state space model
- [[s-mamba|S-Mamba]] — first Mamba MTSF baseline (forecasting, not diffusion)
- [[csdi|CSDI]] — the Transformer-based predecessor
- [[diffusion-model|Diffusion Models]]
- [[ddpm|DDPM]]

[^src-ssdts]: [[source-ssdts]]
