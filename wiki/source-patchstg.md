---
title: "PatchSTG — Efficient Large-Scale Traffic Forecasting with Transformers"
type: source-summary
tags:
  - traffic-forecasting
  - spatial-temporal
  - transformer
  - large-scale
  - kdtree
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# PatchSTG — Efficient Large-Scale Traffic Forecasting with Transformers

Yuchen Fang, Yuxuan Liang, Bo Hui, Zezhi Shao, Liwei Deng, Xu Liu, Xinke Jiang, Kai Zheng. *KDD 2025*. [Code: github.com/LMissher/PatchSTG](https://github.com/LMissher/PatchSTG)

## Core Contribution

PatchSTG proposes an efficient Transformer framework for large-scale traffic forecasting from a **spatial data management perspective**. The key insight: dynamic spatial attention in traffic Transformers has quadratic complexity O(N²d), making it impractical for thousands of sensors. PatchSTG reduces the number of points in attention calculations by borrowing the **patching** idea from vision Transformers — but adapted for **irregularly distributed** traffic points via KDTree-based spatial partitioning[^src-patchstg].

## Method

Four components[^src-patchstg]:

1. **Spatio-Temporal Embedding**: Fully-connected layer transforms raw traffic flow → dₑ-dim embeddings, concatenated with day-of-week, timeslice-of-day, and learnable spatial embedding dictionaries.

2. **Irregular Spatial Patching**: A novel **leaf KDTree** (variant of classical KDTree that ensures all points land in leaf nodes) recursively partitions traffic points by latitude/longitude into balanced, non-overlapping leaf nodes (capacity C). Unfull leaf nodes are padded with temporally-similar points (cosine similarity). Leaf nodes from the same subtree are merged via backtracking into equal-occupancy patches. Complexity: O(N log N) preprocessing.

3. **Dual Attention Encoder**: (a) **Depth Attention** — multi-head self-attention *within* each patch to capture local spatial correlations among geographically close points. (b) **Breadth Attention** — multi-head self-attention *across* patches at the same index position to learn diverse global spatial knowledge without compression. L layers interleaved.

4. **Projection Decoder**: Unpatch (DFS per root node), unpad, then FC layer projects to future F time steps.

Training: L1 loss, AdamW (lr=0.002, weight decay=1e-4), 50 epochs, halved at epochs 2/35/40.

## Complexity

Dominant complexity is O(max(P,R)·M·d) where P (points per patch) ≪ N, R (number of patches) ≪ N, M ≈ N. This is substantially cheaper than dot-product dynamic modeling O(N²d)[^src-patchstg].

## Results

Evaluated on LargeST benchmark (SD: 716, GBA: 2,352, GLA: 3,834, CA: 8,600 nodes; 2019 full year, 5-min granularity, 12→12 forecasting). Compared against 10 baselines: STID (non-spatial), GWNET/AGCRN/STGODE/RPMixer (static spatial), DSTAGNN/D2STGNN/DGCRN/STWave/BigST (dynamic spatial)[^src-patchstg].

- **SOTA** on all four datasets across MAE/RMSE/MAPE (horizon 3/6/12).
- **10× training speedup** and **4× GPU memory reduction** vs D2STGNN/DSTAGNN on CA (8,600 nodes).
- Leaf KDTree is the most critical component: ablation "w/o LKDT" causes large performance drop.
- Dual attention essential: removing depth or breadth both degrade results.
- Cosine-similarity padding outperforms zero-padding and distance-based padding.
- METIS (balanced graph partition) and KMeans alternatives perform worse due to lack of recursive merge and balanced non-overlap guarantees.

## Key Insights

1. **Spatial distinguishability (STID)** alone outperforms spatial propagation methods on large-scale datasets like CA — over-smoothing in message passing hurts more than missing spatial interactions.
2. **Interpretability**: Leaf KDTree explicitly maps real-world adjacent points to same leaf nodes; learned patch-level correlations in breadth attention are diverse per point index (fidelity).
3. Rich patterns in big data mean **small models suffice** — best dimension is 64–128 even for CA.
4. Optimal patch count correlates positively with dataset size (SD: 16, GBA: 16, GLA: 64, CA: 512).

## Limitations

- Only tested on traffic data; extension to other spatio-temporal tasks (e.g., air quality) is future work.
- Patches are fixed at pre-processing; no adaptive re-patching during training.
- Padding with similar points may introduce subtle temporal artifacts that the cosine-similarity heuristic cannot fully eliminate.

[^src-patchstg]: [[source-patchstg]]
