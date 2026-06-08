---
title: "UoMo: A Universal Model of Mobile Traffic Forecasting for Wireless Network Optimization"
type: source-summary
tags:
  - mobile-traffic
  - foundation-model
  - diffusion-model
  - spatio-temporal
  - network-optimization
  - kdd-2025
created: 2026-06-08
last_updated: 2026-06-08
source_count: 0
confidence: high
status: active
---

# UoMo: A Universal Model of Mobile Traffic Forecasting for Wireless Network Optimization

**Authors**: Haoye Chai, Shiyuan Zhang, Xiaoqian Qi (Tsinghua University, BNRist), Baohua Qiu (China Mobile), Yong Li (Tsinghua University)
**Venue**: KDD 2025 ADS Track, August 3-7, Toronto
**Code**: <https://github.com/tsinghua-fib-lab/UoMo>
**arXiv**: 2410.15322

## Core Contribution

UoMo is the **first universal model for mobile traffic forecasting** in wireless networks. It unifies three distinct forecasting tasks — short-term prediction, long-term prediction, and distribution generation — under a single framework, deployed in production on China Mobile's [[jiutian-platform|Jiutian platform]].

## Architecture

UoMo uses a **transformer-based diffusion model** backbone (inspired by Sora/DiT) with three stages:

1. **Data tokenization**: Decomposes mobile traffic data with varying spatial-temporal spans into unified mobile tokens. This handles heterogeneous collection granularities (15-min to 1-hour, regional to cell-level).

2. **Masked diffusion pre-training**: Self-supervised training with 4 task-oriented masking strategies:
   - Short-term mask: masks future portion (e.g., 48-to-16 steps)
   - Long-term mask: masks large future span (e.g., 16-to-48 steps)
   - Generation mask: masks entire temporal dimension at spatial locations for zero-history generation
   - Random mask: random spatio-temporal masking to capture general dependencies

3. **Urban context-aware fine-tuning**: Contrastive learning aligns mobile traffic features with contextual features (mobile users + dynamic POI embeddings). A dynamic POI transformation scheme encodes static POI distributions with temporal indicators, making them time-aware. The contrastive objective is proven equivalent to minimizing InfoNCE loss via the diffusion training process.

Adaptive conditioning (FiLM-style) reshapes transformer LayerNorm scale/shift parameters based on conditional observations.

## Key Results

**Multi-task forecasting (7 cities, 13 baselines)**:
- Long-term prediction: avg RMSE/MAE improvement of **27.85%** vs best baseline
- Short-term prediction: avg improvement of **18.57%**
- Generation task: avg JSD/MAE improvement of **~15.6%**

**Zero/few-shot learning** (Munich, Hangzhou, Fuyang, Nanning):
- UoMo zero-shot surpasses Open-Diff after small-scale training on Munich
- 5-10% few-shot data enables accurate forecasting on unseen cities

**Scaling properties**: Model performance follows log-linear scaling with parameters and data size. Diminishing returns beyond 100M parameters at fixed data size. Optimal size depends on available data volume.

**Ablation**: Removing mobile user features degrades performance by 67-89% (RMSE); removing POI features degrades by 20-48%. Both contextual features are critical.

## Deployment (China Mobile Jiutian Platform)

Live deployment in Nanning, Guangxi Province:
- **BS deployment optimization**: +25.3% served users, -18.03% operation cost, -9.00% capacity deficits
- **BS sleep control**: +21.9% QoS improvement, **-40.7% equipment depreciation**

Trained on 4x NVIDIA A100 (80GB). Model variants: 5M-200M parameters, 12-20 Transformer layers.

## Limitations

- One-for-all design trades task-specific optimization for universality
- Scaling analysis shows optimal model size depends on available data volume; simply increasing parameters does not guarantee better performance
- Current deployment limited to one province; large-scale multi-province deployment pending

## Related Wiki Pages

- [[uomo]] — UoMo model entity page
- [[mobile-traffic-forecasting]] — mobile traffic forecasting domain
- [[jiutian-platform]] — China Mobile's AI platform
- [[masked-diffusion-pre-training]] — task-oriented masking technique
- [[contrastive-diffusion-alignment]] — context-aware contrastive fine-tuning
- [[traffic-forecasting]] — general traffic forecasting concept
- [[spatio-temporal-foundation-model]] — ST foundation model landscape
