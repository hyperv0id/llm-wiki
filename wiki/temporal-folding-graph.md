---
title: "Temporal Folding Graph"
type: technique
tags:
  - tokenization
  - spatial-temporal
  - traffic-forecasting
  - long-term-forecasting
  - representation-learning
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Temporal Folding Graph (TFG)

The Temporal Folding Graph (TFG) is a novel input tokenization strategy for spatial-temporal data, introduced by [[visifold|VisiFold]]. It replaces the conventional spatial-temporal graph representation — which uses a sequence of discrete snapshots — with a single graph where each node encodes its entire temporal history[^src-visifold].

## Motivation

The conventional spatial-temporal graph paradigm has two inherent limitations for long-term forecasting[^src-visifold]:

1. **Snapshot-stacking inflation** — resource overhead (GPU memory, runtime) grows linearly with the number of time steps T and quadratically with the number of nodes N
2. **Cross-step fragmentation** — temporal dependencies are partitioned across separate snapshots, requiring intermediate representations and cross-step message passing, which degrades long-horizon prediction quality

Both stem from the spatial-temporal decoupling inherent in snapshot-based representations[^src-visifold].

## Mechanism

Given input $X_{t-T+1:t} \in \mathbb{R}^{N \times T \times C}$ (N nodes, T time steps, C channels), the TFG representation for the n-th node is[^src-visifold]:

$$X_n^{TF} = \text{Squeeze}(X_{t-T+1:t})[n] \in \mathbb{R}^{1 \times T}$$

Each node token now contains T attributes — the entire temporal window compressed into a single vector. Temporal dynamics are modeled **within** nodes, while spatial dependencies are exchanged **across** nodes on a single graph[^src-visifold].

## Key Properties

- **No temporal module needed** — eliminates the dedicated temporal encoder (RNN/TCN/attention) entirely, as temporal information is embedded in the token representation
- **Eliminates cross-step propagation** — information flows only once on a single graph, not through T sequential snapshots
- **Reduces token count from N×T to N** — an order of magnitude reduction
- **Tokenization-level innovation** — analogous to ViT's pixel-to-patch transformation in CV, but for temporal dimension in spatio-temporal data[^src-visifold]

## Complexity

Time and space complexity both reduce from $O(N \cdot g(T) + T \cdot h(N))$ to $O(h(N))$, where g(T) is the temporal module complexity and h(N) is the spatial module complexity[^src-visifold].

## Comparison: TFG vs. SF

A symmetric alternative — Spatial Folding (SF), collapsing along the spatial dimension — is significantly worse[^src-visifold]. The degradation stems from the inability to seamlessly add spatial embeddings, which are the dominant accuracy driver (per VisiFold's embedding ablation). TFG preserves both inter-node modeling for spatial dependencies and intra-node modeling for temporal correlations[^src-visifold].

## Related Pages

- [[visifold|VisiFold]] — the model that introduced TFG
- [[node-visibility]] — complementary mechanism reducing spatial bottleneck
- [[traffic-forecasting]] — application domain
- [[patch-based-tokenization]] — analogous tokenization concept in general time series

[^src-visifold]: [[source-visifold]]