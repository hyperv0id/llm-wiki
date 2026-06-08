---
title: "SSD-TS — Exploring the Potential of Linear State Space Models for Diffusion Models in Time Series Imputation"
type: source-summary
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
confidence: high
status: active
---

# SSD-TS: Linear State Space Models for Diffusion in Time Series Imputation

**SSD-TS** (State Space Diffusion for Time Series) is a KDD 2025 paper by Hongfan Gao et al. (ECNU) that explores the potential of the [[mamba|Mamba]] selective state space model as the backbone for diffusion-based time series imputation. The paper proposes a Mamba-based noise prediction module within the [[ddpm|DDPM]] conditional diffusion framework, replacing the Transformer attention backbone used in prior work like [[csdi|CSDI]] and the S4 backbone in SSSD [^src-ssdts].

## Core Contributions

1. **Mamba as Diffusion Denoising Backbone**: First work to replace Transformer/S4 backbones with Mamba in a conditional diffusion model for time series imputation. Mamba provides linear time/space complexity $O(NCL)$ vs. Transformer's $O(CL^2)$ and offers superior noise-signal discrimination and controllable frequency response [^src-ssdts].

2. **BAM (Bidirectional Attention Mamba)**: A bidirectional Mamba module with integrated temporal attention for intra-channel dependency modeling across multiple time ranges. The bidirectional scanning ensures each time step captures both forward and backward context, while temporal attention enhances multi-range dependency capture [^src-ssdts].

3. **CMB (Channel Mamba Block)**: A unidirectional Mamba module operating on the channel dimension for inter-channel dependency modeling. Uses transpose operations to reorient the Mamba scan across channels, with iterative hidden state updates that describe dynamic dependencies more accurately than static attention [^src-ssdts].

4. **PNM (Parallel Mamba Block)**: The basic building block replacing Transformer self-attention, consisting of parallel forward/backward convolution + SSM scans, followed by gating, normalization, and residual connections [^src-ssdts].

## Architecture

The noise prediction network consists of input SMM blocks → conditional SMM blocks (where observed values are merged as conditions via zero-padding and conditional masks), with each SMM (Sequential Mamba Module) composed of stacked BAM (intra-channel) + CMB (inter-channel) blocks [^src-ssdts].

## Two Advantages over Attention Backbones

1. **Better Noise-Signal Discrimination**: During early diffusion stages when data is mostly noise, attention weights (computed via input similarity) can be misleading. SSMs update parameters independent of content similarity, with gating mechanisms filtering noise [^src-ssdts].

2. **Controllable Frequency Response**: SSMs offer controllable frequency responses $\hat{K}(\omega) = C(i\omega I - A)^{-1}B$, allowing them to suppress broadband noise while preserving signal in specific frequency bands. Attention mechanisms lack this frequency selectivity [^src-ssdts].

## Results

- **MuJoCo** (90% missing): MSE $6.5\times10^{-4}$, 65.8% improvement over second-best (SSSD) [^src-ssdts]
- **PhysioNet** (all 3 missing rates): Best RMSE, improvements of 22.6%/17.2%/23.5% at 10%/50%/90% [^src-ssdts]
- **CRPS**: Best in 3/4 tasks (10%/50% PhysioNet + AQI), demonstrating superior probabilistic calibration [^src-ssdts]
- **Efficiency**: 87.57M params but faster inference (0.93s vs Transformer's 1.51s) with lower GPU memory (4536MB vs 11250MB) [^src-ssdts]
- **Ablation**: All three components (BAM bidirectional, temporal attention, CMB inter-channel) are essential; temporal attention has largest impact [^src-ssdts]
- **Block missing**: Also evaluated on PTB-XL with 20% block missing, outperforming SSSD [^src-ssdts]
- **Forecasting**: Applicable to forecasting (ETTm1, ranks 1st in 3/4 metrics among baselines) [^src-ssdts]

## Limitations

- Single backbone (only Mamba evaluated, no comparison with Mamba-2 or other SSM variants)
- $C=128$ channel cap due to single-GPU memory; larger channels could improve metrics [^src-ssdts]
- Code location: https://github.com/decisionintelligence/SSD-TS

## Related Pages

- [[ssd-ts|SSD-TS]]
- [[bam|BAM (Bidirectional Attention Mamba)]]
- [[cmb|CMB (Channel Mamba Block)]]
- [[csdi|CSDI]]
- [[mamba|Mamba]]
- [[s-mamba|S-Mamba]]
- [[ddpm|DDPM]]

[^src-ssdts]: [[source-ssdts]]
