---
title: "Over-Smoothing in GNNs"
type: concept
tags:
  - graph-neural-network
  - spectral-methods
  - deep-learning-theory
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Over-Smoothing in GNNs

Over-smoothing is a fundamental limitation of graph neural networks where node representations become increasingly indistinguishable as the number of GNN layers increases, because repeated message passing acts as a low-pass filter that averages away discriminative features[^src-hifinet].

## Mechanism

Most GNNs (GCN, GAT, GraphSAGE) aggregate features from local neighborhoods at each layer. Nt & Maehara (2019) showed that these models inherently perform **low-pass filtering** on graph signals[^src-hifinet]. With stacking, all node features converge toward a common value—erasing the high-frequency variations that encode local structural differences.

Formally, if a GNN's message-passing operation corresponds to multiplication by a low-pass filter matrix $F$, repeated application yields $F^k X$, which converges to the dominant eigenvector (the all-ones vector for connected graphs) as $k$ grows[^src-hifinet].

## Consequences

Over-smoothing is particularly problematic for[^src-hifinet]:

- **Road networks**: High-frequency edges (inner ring roads, intersections) encode critical local traffic variations. Smoothing erases these patterns while preserving coarse commuting trends.
- **Deep GNNs**: Standard GCN performance degrades beyond 2–3 layers, limiting the model's depth and receptive field.
- **Heterogeneous graphs**: Nodes with different characteristics get homogenized, losing discriminative power for classification tasks.

## Mitigation Strategies

| Strategy | Method | Mechanism |
|----------|--------|-----------|
| Skip connections | JK-Net, GCNII | Mix shallow (local) and deep (global) features |
| Frequency-aware GNNs | FAGCN, ACM-GCN | Adaptively balance low- and high-pass filtering |
| Hierarchical modeling | DiffPool, HRNR, [[hifinet|HiFiNet]] | Different resolutions at different hierarchy levels |
| Graph transformers | GT, Graphormer | Global attention bypasses iterative smoothing |
| **Frequency decomposition** | [[hifinet|HiFiNet]] | Explicitly model low- and high-frequency components, reconstruct fused representation |

## HiFiNet's Solution

[[hifinet|HiFiNet]] (AAAI 2026) addresses over-smoothing through a structural solution[^src-hifinet]:

1. **Hierarchy as spectral filter**: Theorem 1 proves that the three-level hierarchy (segment → locality → region) naturally separates frequencies—top-down unpooling acts as a low-pass filter, preserving smooth global patterns.
2. **Explicit high-frequency modeling**: By subtracting low-frequency features from the original, HiFiNet explicitly preserves high-frequency variations that would otherwise be smoothed away.
3. **TGT fusion**: The [[topology-aware-graph-transformer|Topology-Aware Graph Transformer]] fuses both components with a learnable balance parameter $\\beta$, allowing task-adaptive emphasis on global vs. local information.

This makes HiFiNet one of the first GNN frameworks to **structurally** (not just numerically) address over-smoothing through hierarchical graph coarsening[^src-hifinet].

[^src-hifinet]: [[source-hifinet]]
