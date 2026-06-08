---
title: "Masked Diffusion Pre-Training"
type: technique
tags:
  - diffusion-model
  - self-supervised-learning
  - pre-training
  - spatio-temporal
  - masking
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# Masked Diffusion Pre-Training

Masked diffusion pre-training is the core self-supervised training paradigm of [[uomo|UoMo]] (KDD 2025), combining **diffusion models** with **task-oriented spatio-temporal masking** to learn universal mobile traffic representations[^src-uomo].

## Design

The approach draws from two inspirations:
- **Diffusion models**: Denoising diffusion probabilistic models ([[ddpm|DDPM]]) as the generative backbone
- **Masked autoencoders**: Self-supervised pre-training via masking and reconstruction (akin to [[mae|MAE]] in vision)

The combination enables the model to: (a) learn diverse task competencies through different masking patterns, and (b) capture both local and global spatio-temporal correlations in a probabilistic framework.

## Data Tokenization

Before masking, heterogeneous mobile traffic data is tokenized into uniform units. Raw data $S \in \mathbb{R}^{H \times V \times T}$ (region height x width x time) is decomposed into mobile tokens $X \in \mathbb{R}^{(H' \times V' \times T') \times (h_0 \times v_0 \times t_0)}$, then embedded via a learnable embedding layer to $C$-dimensional hidden features[^src-uomo].

## Task-Oriented Masking Strategies

UoMo defines 4 masking strategies corresponding to its 3 forecasting tasks, plus random masking for general feature learning[^src-uomo]:

| Mask | Pattern | Task |
|------|---------|------|
| Short-term | Mask $t_0:T'$ temporally at specific spatial locations | Short-term prediction |
| Long-term | Mask $t_0:T'$ with small $t_0 \ll T'$ | Long-term prediction |
| Generation | Mask entire $0:T'$ temporally at spatial locations | Distribution generation |
| Random | Random spatio-temporal masking | General dependencies |

For prediction masks, the ratio of unmasked to masked time steps determines short vs. long-term prediction capability. For the generation mask, the complete temporal obfuscation forces the model to learn spatial dependencies from surrounding unmasked areas.

## Diffusion Process

After masking, the original tokens are split:
- **Unmasked observations** $o = E_x(X) \odot (1-m)$ → fed as conditioning to the denoising network
- **Masked targets** $e = E_x(X) \odot m$ → corrupted with noise via the forward diffusion process

The denoising network (transformer-based) aims to predict the added noise $\epsilon_\theta(e_k, k|o)$ conditioned on observations $o$, with loss focused only on masked regions[^src-uomo]:

$$L_\theta = \min_\theta \mathbb{E}_{e \sim q(e)} \left[ \|\epsilon - \epsilon_\theta(e_k, k|o)\|^2 \odot m \right]$$

## Adaptive Conditioning

FiLM-style adaptive conditioning is used to inject conditional observations into the transformer backbone. Scale ($\beta$), shift ($\gamma$), and residual ($\alpha$) parameters are predicted from conditions and applied to LayerNorm outputs[^src-uomo]:

$$\alpha, \beta, \gamma = F_\theta(o), \quad e_k \leftarrow e_k + \alpha \cdot A_\theta(\beta e_k + \gamma)$$

This is computationally more efficient than cross-attention conditioning, proven effective in prior work (DiT, Peebles & Xie 2023)[^src-uomo].

## Comparison to Related Pre-Training Paradigms

| Paradigm | Representation | Example |
|----------|---------------|---------|
| Masked diffusion (UoMo) | Task-oriented masks + denoising | [[uomo|UoMo]] |
| MAE pre-training | Spatial-temporal masking + autoencoding | [[gpt-st|GPT-ST]], [[std-mae|STD-MAE]] |
| Diffusion-only | Full sequence denoising | [[d3vae|GCRDD]], [[diffstg|DiffSTG]] |

The key advantage of masked diffusion: a single pre-trained model acquires **all three task capabilities** through different masks, rather than training separate models per task.

[^src-uomo]: [[source-uomo]]
