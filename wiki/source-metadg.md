---
title: "MetaDG — Meta Dynamic Graph for Traffic Flow Prediction"
type: source-summary
tags:
  - traffic-forecasting
  - spatial-temporal
  - graph-neural-network
  - dynamic-graph
  - meta-learning
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# Source: MetaDG (AAAI 2026)

**Full title**: Meta Dynamic Graph for Traffic Flow Prediction  
**Authors**: Yiqing Zou, Hanning Yuan, Qianyu Yang, Ziqiang Yuan, Shuliang Wang, Sijie Ruan (Beijing Institute of Technology)  
**Venue**: AAAI 2026  
**arXiv**: [2601.10328](https://arxiv.org/abs/2601.10328)  
**Code**: [github.com/zouyiqing-221/MetaDG](https://github.com/zouyiqing-221/MetaDG)

## Summary

MetaDG is a GCRU-based spatio-temporal prediction framework that simultaneously models **dynamics** and **heterogeneity** for traffic flow forecasting[^src-metadg]. The paper identifies two key limitations of existing work: (1) dynamics modeling is typically restricted to spatial topology changes (e.g., dynamic adjacency matrices), whereas dynamics could operate at a broader scale; (2) spatio-temporal heterogeneity is modeled separately for spatial and temporal dimensions rather than jointly[^src-metadg].

MetaDG addresses these through three novel modules:
1. **Dynamic Node Generation (DNG)** — generates raw dynamic node embeddings at each time step via a time-gated fusion of static node embeddings and hidden states, determining the strength of dynamics per dimension[^src-metadg].
2. **Spatio-Temporal Correlation Enhancement (STCE)** — enhances node embeddings through spatial cross-attention (SCE, extracting global historical information) followed by temporal smoothing (TCE, using GRU update gates to stabilize across time steps), in a fusion-before-smoothing order[^src-metadg].
3. **Dynamic Graph Qualification (DGQ)** — refines the dynamically generated adjacency matrix by qualifying the reliability of message-passing on edges, using cross-time-step similarity to produce adaptive scaling coefficients for proportional edge strengthening/weakening[^src-metadg].

These modules feed into the **Meta-DGCRU**, which generates meta-parameters (node-wise model weights), raw adjacency matrices, and edge-weight adjustment matrices at each time step — replacing the static parameters and graph of standard GCRU[^src-metadg].

## Key Results

SOTA on PEMS03/04/07/08 across all metrics (MAE/RMSE/MAPE)[^src-metadg]:
- PEMS03: MAE 14.29, RMSE 24.93, MAPE 14.64% (vs ST-SSDL 14.56/25.79/15.08%)
- PEMS08: MAE 13.04, RMSE 22.53, MAPE 8.58% (vs HimNet 13.57/23.25/8.99%)

Ablation shows all components matter: removing STCE, DGQ, reversing the SCE-TCE order (TSCE), or using a shared (Joined) embedding all degrade performance[^src-metadg]. MetaDG shows particular advantage in long-term predictions and achieves comparable inference time to ST-SSDL with fewer parameters than HimNet[^src-metadg].

## Contributions
- Extends dynamic modeling beyond adjacency matrices to meta-parameters, pushing ST-isolated models toward ST-unification
- Proposes DGQ for qualifying message-passing reliability in GCRU-based models
- Unifies spatio-temporal heterogeneity modeling into a single dynamic framework

[^src-metadg]: [[source-metadg]]
