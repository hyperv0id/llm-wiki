---
title: "Over-Smoothing in GNNs"
type: concept
tags:
  - graph-neural-network
  - spectral-methods
  - deep-learning-theory
created: 2026-06-08
last_updated: 2026-08-30
source_count: 4
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

## 与 Over-Squashing 的区分

Over-smoothing 不是长程性能退化的唯一候选解释，也不应与 [[over-squashing|over-squashing]] 混用——二者是不同现象：over-smoothing 指层数增加后节点表示趋同、不可区分；over-squashing 指指数增长的感受野信息被压缩进固定长度向量，发生在需要长程交互的任务中。Alon & Yahav（ICLR 2021）指出 over-smoothing 的实证证据主要来自短程任务，并以假设口径提出：在长程问题上，性能退化的解释是 over-squashing 而非 over-smoothing（Sec 1, Sec 6）[^src-over-squashing]。该文用两个构造性例子说明二者可独立发生：三节点完全图（问题半径 r=1）上可能出现 over-smoothing 而无 over-squashing；Tree-Neighbors-Match 任务上存在 over-squashing 而无 over-smoothing（Appendix E）[^src-over-squashing]。

## Mitigation Strategies

| Strategy | Method | Mechanism |
|----------|--------|-----------|
| Skip connections | JK-Net, GCNII | Mix shallow (local) and deep (global) features |
| Frequency-aware GNNs | FAGCN, ACM-GCN | Adaptively balance low- and high-pass filtering |
| Hierarchical modeling | DiffPool, HRNR, [[hifinet|HiFiNet]] | Different resolutions at different hierarchy levels |
| Graph transformers | GT, Graphormer, [[graphgps\|GPS]] | Global attention lets information spread via full connectivity, bypassing iterative smoothing[^src-hifinet][^src-graphgps] |
| **Frequency decomposition** | [[hifinet|HiFiNet]] | Explicitly model low- and high-frequency components, reconstruct fused representation |
| **Sheaf Laplacian** | [[ssf|SSF]] | Replace graph Laplacian with [[sheaf-laplacian|sheaf Laplacian]] — edge-specific restriction maps prevent uniform feature averaging, preserving local discriminative signals[^src-ssf] |
| **Spectral filtering** | [[ssf|SSF]] | Heat kernel $e^{-\alpha\lambda}$ over sheaf Laplacian eigenspectrum suppresses high-frequency noise while retaining low-frequency structure[^src-ssf] |

## HiFiNet's Solution

[[hifinet|HiFiNet]] (AAAI 2026) addresses over-smoothing through a structural solution[^src-hifinet]:

1. **Hierarchy as spectral filter**: Theorem 1 proves that the three-level hierarchy (segment → locality → region) naturally separates frequencies—top-down unpooling acts as a low-pass filter, preserving smooth global patterns.
2. **Explicit high-frequency modeling**: By subtracting low-frequency features from the original, HiFiNet explicitly preserves high-frequency variations that would otherwise be smoothed away.
3. **TGT fusion**: The [[topology-aware-graph-transformer|Topology-Aware Graph Transformer]] fuses both components with a learnable balance parameter $\\beta$, allowing task-adaptive emphasis on global vs. local information.

This makes HiFiNet one of the first GNN frameworks to **structurally** (not just numerically) address over-smoothing through hierarchical graph coarsening[^src-hifinet].

## SSF's Sheaf-Based Solution

[[ssf|SSF]] (ICLR 2026, under review) addresses over-smoothing through a fundamentally different approach — replacing the graph Laplacian entirely[^src-ssf]:

1. **Sheaf Laplacian vs. Graph Laplacian**: Standard GNN diffusion uses the normalized graph Laplacian, which converges all node features to the same value. The [[sheaf-laplacian|sheaf Laplacian]] $L_F$, by incorporating edge-specific restriction maps $F_{v \triangleleft e}$, ensures that feature transformations are **directionally and contextually differentiated** — information does not simply average out[^src-ssf].
2. **Spectral filtering with heat kernel**: SSF decomposes $L_F = U\Lambda U^T$ and applies $e^{-\alpha\lambda}$ in the spectral domain. This selectively **suppresses high-frequency noise** (which contributes to instability) while preserving low-frequency structural patterns — unlike spatial GNNs where all frequencies are mixed indiscriminately[^src-ssf].

The combined effect: restriction maps provide local protection against over-smoothing, while spectral filtering provides global frequency control. SSF's strong long-horizon performance (60-min MAE degrades only ~80% from 15-min, vs. ~200% for baselines) empirically validates this approach[^src-ssf].

[^src-hifinet]: [[source-hifinet]]
[^src-ssf]: [[source-ssf]]
[^src-graphgps]: [[source-graphgps]]
[^src-over-squashing]: [[source-over-squashing]]
