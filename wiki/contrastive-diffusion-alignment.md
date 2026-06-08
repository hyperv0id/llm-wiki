---
title: "Contrastive Diffusion Alignment"
type: technique
tags:
  - contrastive-learning
  - diffusion-model
  - fine-tuning
  - InfoNCE
  - mobile-networks
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# Contrastive Diffusion Alignment

Contrastive Diffusion Alignment is the urban context-aware fine-tuning strategy in [[uomo|UoMo]] (KDD 2025), which uses a **contrastive learning approach embedded within the diffusion training process** to align mobile traffic features with contextual urban features (mobile users, POI distributions)[^src-uomo].

## Motivation

Mobile traffic is not purely a spatio-temporal sequence — it is fundamentally shaped by **urban contexts**[^src-uomo]:
- **Mobile user counts**: Dynamically reflect human mobility patterns
- **POI distributions**: Statically encode urban functional layout, but their influence varies by time (e.g., restaurants peak at lunch/dinner)

UoMo's fine-tuning stage integrates these contextual signals to improve both forecasting accuracy and cross-city transferability.

## Contextual Data Transformation

Two types of contextual data are processed[^src-uomo]:

1. **Mobile users**: Tokenized with the same tokenization as traffic data, yielding $c_u \in \mathbb{R}^{(H' \times V' \times T') \times (h_0 \times v_0 \times t_0)}$. Can be directly input into the network.

2. **Dynamic POI embeddings**: Static POI vectors ($P \in \mathbb{R}^{H \times V}$) are transformed into time-aware representations through a two-step process:
   - Static feature extraction: $h_{sp} = \sigma(W_s \cdot P + B_s)$
   - Temporal injection: $h_{dp} = \sigma(W_l \cdot [h_{sp} \oplus \tau(t)] + B_l)$, where $\tau(t)$ is a 2D temporal embedding (day, hour) projected via MLP

The final contextual token is $y = c_u + c_p$ (mobile user + dynamic POI features).

## Contrastive Learning via Diffusion

The core insight: **training a diffusion model with positive and negative sample pairs is equivalent to minimizing the InfoNCE loss** in contrastive learning (Lemma 1, proven in appendix A.3 of the paper)[^src-uomo].

- **Positive samples**: Mobile traffic tokens and contextual feature tokens from the **same spatio-temporal block**
- **Negative samples**: Tokens from **different spatio-temporal blocks**

The training objective becomes[^src-uomo]:

$$L \approx \mathbb{E}\left[ \|\epsilon - \epsilon_\theta(e, k|y)\|^2 - \lambda \sum_{e'} \|\epsilon - \epsilon_\theta(e', k|y)\|^2 \right] \odot m$$

This simultaneously:
1. Minimizes denoising error for positive (matching) traffic-context pairs
2. Maximizes denoising error for negative (mismatched) pairs, scaled by $\lambda \propto \log N$

The equivalence to InfoNCE implies the objective maximizes mutual information $I(e, y)$ between traffic features $e$ and contextual features $y$[^src-uomo].

## Fine-Tuning Strategy

During fine-tuning, most pre-trained parameters (attention layers, linear layers, MLP networks) are **frozen** to preserve general spatio-temporal features. Only the adaptive conditioning layers and output decoder are updated, minimizing computational cost[^src-uomo].

## Ablation Results

Removing contextual features causes severe degradation[^src-uomo]:

| Removed Module | RMSE Degradation (Prediction) | JSD Degradation (Generation) |
|---------------|-------------------------------|------------------------------|
| Mobile users (UoMo w/o User) | -67.47% to -89.61% | -63.13% to -76.53% |
| POI distribution (UoMo w/o POI) | -19.81% to -47.75% | -21.77% to -34.69% |

Mobile user dynamics are the most critical contextual feature, reflecting the dominant role of human mobility patterns in mobile traffic.

[^src-uomo]: [[source-uomo]]
