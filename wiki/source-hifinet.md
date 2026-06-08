---
title: "HiFiNet: Hierarchical Frequency-Decomposition Graph Neural Networks for Road Network Representation Learning"
type: source-summary
tags:
  - graph-neural-network
  - road-network
  - frequency-decomposition
  - hierarchical-modeling
  - representation-learning
  - spectral-methods
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# source-hifinet

**HiFiNet** is a hierarchical frequency-decomposition GNN for road network representation learning, proposed by Ma, Wang (Beihang University), and U (University of Macau), accepted at AAAI 2026[^src-hifinet].

## Core Contribution

HiFiNet unifies spatial and spectral modeling for road networks through two key innovations[^src-hifinet]:

1. **Three-level hierarchy**: road segments → localities (e.g., intersections) → regions (e.g., residential/commercial zones), with learnable cross-attention-based assignment matrices that capture multi-scale spatial semantics and enable localized frequency analysis.
2. **Frequency-decomposition learning**: a decomposition–updating–reconstruction paradigm that explicitly separates low-frequency (smooth global patterns) and high-frequency (local variations) graph signals, processes them via a Topology-Aware Graph Transformer (TGT), and fuses the enriched components.

HiFiNet theoretically proves that the hierarchical projection acts as a spectral low-pass filter, naturally separating frequency components and mitigating over-smoothing[^src-hifinet].

## Key Technical Details

- **Hierarchy construction**: Contextual segment embeddings (ID, lane number, length, geolocation) → learnable segment-to-locality and locality-to-region assignment via cross-attention softmax → top-down low-frequency propagation via GAT unpooling.
- **Frequency decomposition**: High-frequency features $H_S^h = H_S - H_S^l$ (original minus low-frequency). Both components updated through TGT.
- **Topology-Aware Graph Transformer (TGT)**: global attention with learnable parameter $\alpha$ blending self-attention and adjacency matrix: $\alpha \cdot \text{softmax}(QK^T/\sqrt{d}) + (1-\alpha) \cdot A_S$.
- **Reconstruction**: $\\hat{H}_S = \\beta \cdot \\tilde{H}_S^l + (1-\\beta) \cdot \\tilde{H}_S^h$, with learnable $\beta$.
- **Training losses**: contrastive alignment loss (child-parent consistency), reconstruction loss, semantic loss (pairwise similarity aligns with topology+OD flow), and entropy loss (sharp assignment distributions).

## Experiments

Three real-world datasets (Beijing, Chengdu, Xi'an) with map-matched taxi trajectory data, evaluated on four tasks[^src-hifinet]:

| Task | Metric | Key Result |
|------|--------|-----------|
| Next Location Prediction | ACC@1/ACC@5 | Best across all datasets |
| Label Classification | F1/AUC | Best, e.g., 0.838 F1 on BJ |
| Destination Prediction | ACC@1/ACC@5 | Best across all datasets |
| Route Planning | F1/EDT | Best across all datasets |

Ablations show hierarchy (NB < NL < NR < full) and frequency decomposition (NLF < NHF < full) both critical. Parameter sensitivity: optimal NL=200, NR=30. t-SNE visualization shows clean separation of road types by frequency components.

## Limitations

The paper focuses exclusively on road networks. Generalization to other graph domains remains unexplored. The three-level hierarchy design assumes specific spatial granularity (segment/locality/region) that may not transfer to arbitrary graphs.

[^src-hifinet]: [[source-hifinet]]
