---
title: "CaST: Causal Spatio-Temporal Neural Network for STG Forecasting"
type: source-summary
tags:
  - causal-inference
  - spatiotemporal
  - ood-generalization
  - 2023
  - neurips
created: 2026-07-07
last_updated: 2026-07-14
source_count: 1
confidence: medium
status: active
---

# CaST: Deciphering Spatio-Temporal Graph Forecasting — A Causal Lens and Treatment

Xia et al. (NUS, HKUST(GZ), BJTU, USTC, NeurIPS 2023) propose **CaST**, a causal framework for spatio-temporal graph (STG) forecasting that simultaneously tackles two fundamental challenges: temporal out-of-distribution (OoD) generalization and dynamic spatial causation modeling[^src-cast].

## Core Causal Framework

The paper first constructs a Structural Causal Model (SCM) for STG data generation, identifying two back-door paths between historical observations X and future signals Y: **X ← E → Y** (temporal environment E as confounder) and **X ← C → Y** (spatial context C as confounder)[^src-cast].

### Back-Door Adjustment for Temporal OoD

To address temporal distribution shift, CaST applies back-door adjustment by stratifying temporal environments into discrete types. An **Environment Disentangler** separates input data into environment features (capturing long-term global information via multi-scale 1D convolutions) and entity features (capturing local/periodic information via FFT + self-attention). A learnable **Vector Quantization codebook** discretizes environments, enabling generalization to unseen environments through soft combination of learned environment vectors. Mutual information minimization between entity and environment representations ensures disentanglement[^src-cast].

### Front-Door Adjustment for Dynamic Spatial Causation

For spatial confounding, CaST uses front-door adjustment with a mediating surrogate variable filtered by genuine causal relations. A novel **Hodge-Laplacian (HL) Deconfounder** performs edge-level spectral convolution using the first-order HL operator (L₁ = ∂₂∂₂ᵀ + ∂₁ᵀ∂₁), approximated via Laguerre polynomial expansion. This captures the **ripple effect** of causal relations — how causal strength between nodes propagates across the graph over time, e.g., an accident at one edge weakening neighboring causal links in subsequent time steps[^src-cast].

## Empirical Results

Experiments on three real-world datasets (PEMS08 traffic, AIR-BJ and AIR-GZ PM2.5) demonstrate:
- Consistent SOTA performance across all datasets, outperforming DCRNN, STGCN, ASTGCN, MTGNN, AGCRN, GMSDR, and STGNCDE
- CaST achieves MAE 16.44 (vs AGCRN 17.06) on PEMS08, 22.90 (vs AGCRN 23.43) on AIR-BJ, 12.36 (vs AGCRN 12.74) on AIR-GZ
- Ablation studies confirm each component (environment, entity, edge convolution) contributes meaningfully
- Edge convolution (HL-based) outperforms both adaptive adjacency matrix and graph attention variants
- Causal heatmap visualizations on Beijing air quality stations demonstrate interpretable dynamic causation aligned with wind direction patterns[^src-cast]

## Significance

CaST is the first work to jointly address temporal OoD and dynamic spatial causation for STG forecasting from a causal perspective. Its interpretable environment codebook reveals meaningful connections to external factors (temperature, pressure), and the HL-based edge convolution provides a principled topological approach to modeling causal ripple effects. The framework has implications for smart city applications including traffic and air quality forecasting[^src-cast].
## 相关页面

- [[source-e2-cstp]] — 因果多模态时空预测
- [[source-causalx]] — CausalX (ICML 2026)，多源因果约束 + 扩散精炼


[^src-cast]: [[source-cast]] — CaST: Deciphering Spatio-Temporal Graph Forecasting — A Causal Lens and Treatment (Xia et al., NeurIPS 2023)
