---
title: "USTD: Towards Unifying Diffusion Models for Probabilistic Spatio-Temporal Graph Learning"
type: source-summary
tags:
  - diffusion-models
  - spatio-temporal-graph
  - probabilistic-forecasting
  - kriging
  - traffic-forecasting
  - air-quality
  - ddpm
  - sigspatial-2024
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: medium
status: active
---

# USTD: Towards Unifying Diffusion Models for Probabilistic Spatio-Temporal Graph Learning

**Authors**: Junfeng Hu, Xu Liu, Zhencheng Fan, Yuxuan Liang (corresponding), Roger Zimmermann
**Affiliation**: National University of Singapore / University of Technology Sydney / HKUST (Guangzhou)
**Venue**: SIGSPATIAL 2024 (arXiv:2310.17360v2, September 2024)
**Code**: https://github.com/hjf1997/USTD

## Core Problem

Spatio-temporal graph learning tasks (forecasting and kriging) are typically addressed by dedicated models with specialized architectures. These deterministic approaches cannot model intrinsic data uncertainties, and their task-specific designs conflict with the trend toward unified solutions [^src-ustd]. USTD proposes to treat both tasks uniformly as conditional distribution modeling $P_{\phi,\theta}(Y|X)$ — conditions $X$ share the same spatio-temporal dependencies, only the generation targets $Y$ differ.

## Key Contributions

### 1. Pre-Trained Spatio-Temporal Encoder

The encoder uses a GWNet-style STGNN backbone (gated 1D conv + GCN + skip connections) and is pre-trained via unsupervised autoencoding with two enhancements [^src-ustd]:

- **Graph Sampling** (80% node subset per iteration): Prevents the encoder from memorizing the full graph structure, essential for the kriging task where observed/target node sets differ ($N:M=2:1$)
- **Masking** (75% rate, MAE-style): Forced extraction of meaningful patterns from sparse signals, prevents trivial identity mapping in the high-dimensional latent space ($d_h=64 \gg d_x$)

TCN does not use zero padding, so temporal dimension is squashed ($\tau \ll T$), producing low-dimensional conditional representations $\mathbf{H} \in \mathbb{R}^{N\times\tau\times d_h}$.

### 2. Task-Specific Denoising Decoders

Conditional DDPM with task-specialized attention networks [^src-ustd]:

- **TGA (Temporal Gated Attention)** for forecasting: Cross-attention on temporal axis (per-node, target vs historical representations) + Self-attention across nodes + Gated fusion. Complexity $O(N(\tau+1+N))$ vs full attention's $O((N(\tau+1))^2)$
- **SGA (Spatial Gated Attention)** for kriging: Temporal dimension absorbed by embedding layer; Cross-attention on spatial axis (target nodes vs observed node representations) + Self-attention among target nodes + Gated fusion. Complexity $O((M+N)M)$ vs full attention's $O(((M+N)\tau)^2)$

Unattended dimensions (spatial for TGA, temporal for SGA) are handled by the encoder, ensuring completeness without redundant computation.

### 3. Decoupled Training Strategy

The encoder and denoiser are trained separately — a critical departure from CSDI/PriSTI/DiffSTG's coupled optimization. Pre-training first establishes high-quality conditional representations; the denoiser then only needs to learn prediction distributions [^src-ustd]. This resolves the well-known "diffusion STG cannot beat deterministic baselines" problem.

## Results

Evaluated on 4 datasets (PEMS-03, PEMS-BAY, AIR-BJ, AIR-GZ) against 16 baselines [^src-ustd]:

- **Forecasting**: USTD surpasses all probabilistic methods and nearly all deterministic baselines. CRPS reduced by up to 12.0% (PEMS-BAY), MAE by up to 2.4% vs best deterministic. The only metric where deterministic holds advantage is PEMS-BAY RMSE (STGODE 3.33 vs USTD 3.55)
- **Kriging**: USTD surpasses ALL baselines. MAE reduced up to 10.5% (AIR-GZ, 8.61 vs IGNNK 9.62), CRPS up to 7.8% (AIR-GZ, 0.213 vs PriSTI 0.231)
- **Inference speed**: 0.49–0.50s per inference — 42.8–47.3% faster than CSDI (~0.88s) and ~2× faster than PriSTI (~1.05s). Speed gains from TCN temporal compression reducing denoising computation

### Ablation Key Findings

- Removing encoder (w/o EN): catastrophic performance collapse
- Removing pre-training (w/o PT, end-to-end training): significant degradation, confirming coupling issue
- Removing masking (w/o MK): performance drops due to trivial latent space solutions
- Removing graph sampling (w/o GS): forecasting unaffected, kriging drops significantly — GS's value is in cross-graph generalization

## Comparative Positioning

| Aspect | CSDI/PriSTI/DiffSTG | USTD |
|--------|---------------------|------|
| Encoder-Denoiser coupling | Coupled (end-to-end) | Decoupled (pre-train then fine-tune) |
| Temporal dimensionality | Full-length | Compressed ($\tau \ll T$) |
| Attention scope | All dimensions | Task-critical dimension only |
| Deterministic baseline surpass? | Generally no (for forecasting) | Yes (except PEMS-BAY RMSE) |

## Limitations

1. Task coverage limited to forecasting and kriging — imputation, classification, anomaly detection not yet unified [^src-ustd]
2. TGA and SGA trained separately — single model cannot do both tasks simultaneously
3. Static adjacency matrix (road network or geographic distance), no dynamic graph learning
4. Single-modal numerical input only, no multimodal conditioning

## Legacy and Influence

USTD provides a reusable template — pre-trained encoder + task-specific denoiser — for spatio-temporal diffusion. The GSM (graph sampling + masking) pre-training strategy is a simple yet effective trick applicable to other ST encoders. USTD's key argument — that diffusion STG can beat deterministic methods if conditions and generation are decoupled in training — overturned the 2023 consensus [^src-ustd].

[^src-ustd]: [[source-ustd]]
