---
title: "Multi-Modality-Guided Spatial Expert"
type: technique
tags:
  - spatial-modeling
  - mixture-of-experts
  - multi-modality
  - spatio-temporal
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# Multi-Modality-Guided Spatial Expert

**Multi-Modality-Guided Spatial Expert** is a technique introduced in [[most|MoST]] (KDD 2026) that uses multi-modality context to dynamically select specialized spatial experts for modeling region-specific spatial dependencies in traffic prediction[^src-most].

## Motivation

Spatial patterns are highly localized and specific to individual regions, making it suboptimal to use a single neural network to model all regions uniformly[^src-most]. However, regions with similar multi-modality background—such as those near schools—often exhibit comparable spatial patterns. This insight motivates using multi-modality data to guide the selection of appropriate spatial experts.

## Architecture

### Spatial Expert Design

Each spatial expert models interactions between a sensor and its top-k nearest neighbors[^src-most]:

1. **Neighborhood selection**: For each sensor i, identify the top-k neighbors with the highest adjacency weights from the graph adjacency matrix A
2. **Reweighting**: Neighbor embeddings are reweighted by their adjacency weights
3. **Cross-attention**: The sensor's time-series patch embeddings serve as queries; reweighted neighbor embeddings serve as keys and values
4. **Feedforward + residual**: The attention output passes through an MLP with residual connection and layer normalization

This localized design avoids the O(N²) complexity of full-graph modeling while capturing the most relevant spatial dependencies.

### Expert Types

Two types of experts are employed[^src-most]:

1. **Modality-shared experts** (N_modality experts): One expert per input modality. For each activated modality (determined by the [[multi-modality-refinement|modality selector]]), the corresponding modality-shared expert SE-M* is activated.

2. **Routed experts** (N_expert experts, default 4): Selected via a router that takes the modality-fused embedding Q as input and computes routing probabilities. The expert with the highest probability is activated.

### Load Balancing

To prevent expert collapse (where all sensors route to the same expert), a balance loss is applied[^src-most]:

$$L_{balance} = N_{expert} \sum_{i=1}^{N_{expert}} \text{Distribute}_i \times \text{Probability}_i$$

This encourages an even distribution of sensors across all routed experts within each training batch, following the approach of Switch Transformers[^src-most].

### Final Fusion

The outputs from all activated modality-shared experts and the selected routed expert are combined via a weighted sum, using probabilities from the selector and router[^src-most].

## Comparison with Other MoE Approaches

| Approach | Model | Expert Type | Routing Signal |
|----------|-------|-------------|----------------|
| **MoST Spatial Expert** | MoST | Spatial cross-attention experts | Multi-modality context |
| HA-MoE | FaST | GLU-based FFN experts | Raw time series + spatial/temporal bias |
| Switch Transformer | — | FFN experts | Token embedding |

MoST's spatial experts are specifically designed for spatial correlation modeling (cross-attention with neighbors), unlike FaST's HA-MoE which uses GLU-based FFN experts for feature transformation[^src-most].

## Related Pages

- [[most]] — the MoST model
- [[multi-modality-refinement]] — modality selection technique
- [[mixture-of-experts]] — general MoE concept
- [[adaptive-graph-agent-attention]] — AGA-Att (FaST's spatial efficiency technique)
- [[spatio-temporal-foundation-model]] — ST foundation model concept

[^src-most]: [[source-most]]