---
title: "source-e2-cstp"
type: source-summary
tags:
  - spatio-temporal
  - multimodal
  - causal-inference
  - mamba
  - traffic-forecasting
  - neurips
created: 2026-06-04
last_updated: 2026-07-17
source_count: 2
confidence: medium
status: active
---

# E²-CSTP: Causal Spatio-Temporal Prediction — An Effective and Efficient Multi-Modal Approach

Huang et al. (Zhejiang University, NeurIPS 2025) propose **E²-CSTP**, a multi-modal spatio-temporal prediction framework that addresses three challenges: insufficient multi-modal fusion, confounding factors obscuring causal relations, and high computational complexity[^src-e2-cstp]. Source: `raw/e2-cstp-huang-2025.pdf`.

## Three Core Components

### 1. Cross-Modal Feature Fusion

Integrates spatio-temporal sequences, event-related text (via BERT), and environmental images (via CNN) using cross-modal attention + adaptive fusion gating[^src-e2-cstp]. Aligns modalities across temporal and spatial dimensions before fusion.

### 2. Dual-Branch Causal Inference

- **Causal matrix construction**: DeepSHAP estimates latent spatial causal dependencies, blended with prior graph via EMA[^src-e2-cstp]
- **Main branch**: Pure ST sequence prediction
- **Auxiliary branch**: Multi-modal fused features → captures bias from external factors
- **Backdoor adjustment**: Formalizes confounding via SCM — intervenes on ST data to block backdoor path Xst ← S → Yst[^src-e2-cstp]
- Loss: `L_all = L_pred + β·L_st + (1-β)·L_mm`

### 3. STED (Spatio-Temporal Encoding and Decoding)

GCN for spatial dependencies + Mamba (selective state space model) for temporal dynamics — **linear O(B·T·N²·d)** complexity vs Transformer's **O(B·T²·N²·d)**[^src-e2-cstp]. Three stacked GCN+Mamba layers with residual connections and LayerNorm.

## Key Results

| Dataset | Modalities | Best Baseline | E²-CSTP | Improvement |
|---------|-----------|:--:|:--:|:--:|
| Terra | ST + image + text | UniST (MAE 2.47) | **2.43** | 1.61% |
| BjTT | Traffic + events | UniST (MAE 3.62) | **3.56** | 1.66% |
| GreenEarthNet | Satellite + vegetation | UniST (MAE 0.14) | **0.13** | 7.14% |
| BikeNYC | ST only | UniST (MAE 3.31) | **2.99** | **9.66%** |

Efficiency: **17.37%–56.11%** reduction in computational overhead vs Transformer baselines[^src-e2-cstp].

## Ablation Highlights

All six components (text, image, DeepSHAP, causal inference, GCN, Mamba) contribute. Removing causal inference leads to reduced robustness, especially under confounding conditions. Removing GCN or Mamba significantly degrades performance across all datasets[^src-e2-cstp].

## Related Work
[[causalx]] (ICML 2026) takes a different causal route for multi-modal ST: instead of SCM-based confounding adjustment, it learns causal-inspired dynamic graphs via multi-source causal constraints (Granger, do-calculus, TDMI, VAE) with diffusion-based refinement[^src-causalx].

E²-CSTP's core causal mechanism is [[backdoor-adjustment]], also employed by CaST (NeurIPS 2023) for temporal environment confounding[^src-e2-cstp].

[^src-e2-cstp]: [[source-e2-cstp]]
[^src-causalx]: [[source-causalx]]
