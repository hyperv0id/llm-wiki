---
title: "Source: GPT-ST"
type: source-summary
tags:
  - spatial-temporal
  - pre-training
  - masked-autoencoder
  - traffic-forecasting
  - hypergraph
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: high
status: active
---

# Source: GPT-ST

**Full title**: GPT-ST: Generative Pre-Training of Spatio-Temporal Graph Neural Networks

**Authors**: Zhonghang Li, Lianghao Xia, Yong Xu, Chao Huang (South China Univ. of Technology / Univ. of Hong Kong / PAZHOU LAB)

**Venue**: NeurIPS 2023

**arXiv**: 2311.04245

## Summary

GPT-ST proposes the first general-purpose spatio-temporal pre-training framework that seamlessly integrates with diverse downstream ST prediction models to enhance their performance[^src-gpt-st]. Unlike prior end-to-end models that rigidly couple all modules, GPT-ST is a plug-and-play pre-training engine that augments existing STGNNs (STGCN, GWN, MTGNN, MSDR, etc.) without modifying their architectures[^src-gpt-st].

The framework is built on three pillars:

1. **Customized temporal pattern encoding** — Uses a learnable temporal hypergraph with time-specific and region-specific parameters generated via a parameter learner (not directly learned), ensuring each region retains its distinctive temporal signature despite message passing[^src-gpt-st].

2. **Hierarchical spatial pattern encoding** — A hypergraph capsule clustering network uses dynamic routing (inspired by capsule networks) to soft-assign regions to cluster centroids, followed by a high-level cross-cluster hypergraph that models inter-cluster migration patterns (e.g., residential → commercial commuting flows)[^src-gpt-st].

3. **Cluster-aware adaptive mask strategy** — Instead of random masking, GPT-ST uses predicted cluster assignments to progressively increase mask difficulty: from intra-cluster random masking (easy) to whole-cluster masking (hard), forcing the model to learn cross-category reasoning[^src-gpt-st].

## Key Results

Evaluated on 4 datasets (PEMS08, METR-LA, NYC Taxi, NYC Citi Bike) across 13 downstream baselines. Key findings[^src-gpt-st]:

- **Universal improvement** — all 13 baselines improved on all 4 datasets across all 3 metrics (MAE, RMSE, MAPE). Zero counterexamples.
- STGCN w/ GPT-ST on PEMS08: MAE 17.85→16.24 (−9.0%), RMSE 28.64→25.93 (−9.5%)
- Outperforms STEP (prior ST pre-training baseline) while using only 12-step input (vs STEP's 2-week data) and training 26× faster (12.5s vs 327.8s/epoch)
- Classic baselines (STGCN, TGCN) benefit more than advanced baselines (MSDR, STWA) — advanced models already encode similar knowledge

## Limitations

- Task-specific pre-training — a model pre-trained on PEMS08 cannot be reused on NYC Taxi due to different data shapes/distributions[^src-gpt-st]
- Fixed cluster count H_S=10 across all cities assumes uniform urban zoning[^src-gpt-st]
- Marginal gains on highly refined baselines (MSDR: −3.3% MAE) raise questions about incremental value vs. pre-training cost[^src-gpt-st]

[^src-gpt-st]: [[source-gpt-st]]
