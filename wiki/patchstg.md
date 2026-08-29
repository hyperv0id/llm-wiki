---
title: "PatchSTG"
type: entity
tags:
  - traffic-forecasting
  - spatial-temporal
  - transformer
  - kdtree
created: 2026-06-08
last_updated: 2026-08-29
source_count: 3
confidence: high
status: active
---

# PatchSTG

PatchSTG is an efficient Transformer framework for large-scale traffic forecasting, published at KDD 2025[^src-patchstg]. It addresses the quadratic complexity bottleneck of dynamic spatial modeling by borrowing the patching paradigm from vision Transformers (ViT, Swin) and adapting it to **irregularly distributed** traffic points via a novel [[leaf-kdtree|leaf KDTree]] spatial partitioning algorithm.

**Authors**: Yuchen Fang, Yuxuan Liang*, Bo Hui, Zezhi Shao, Liwei Deng, Xu Liu, Xinke Jiang, Kai Zheng* (UESTC, HKUST-GZ, Auburn, ICT-CAS, NUS, PKU)[^src-patchstg].

**Code**: [github.com/LMissher/PatchSTG](https://github.com/LMissher/PatchSTG)

## Architecture

Four-stage pipeline[^src-patchstg]:

1. **Spatio-Temporal Embedding**: FC projection + day-of-week, timeslice-of-day, learnable spatial identity embeddings
2. **[[irregular-spatial-patching|Irregular Spatial Patching]]**: [[leaf-kdtree|Leaf KDTree]] → cosine-similarity padding → subtree backtracking → balanced, non-overlapping patches
3. **Dual Attention Encoder**: Depth attention (within-patch local) ↔ Breadth attention (across-patch global), L=5 layers
4. **Projection Decoder**: DFS un-patching → un-pad → FC layer → future prediction

## Key Innovations

- **First to bridge KDTree and Transformer patching** for irregular spatial data[^src-patchstg].
- **Dual attention** (depth + breadth) captures both local and global spatial dependencies without low-rank compression, preserving interpretability and fidelity[^src-patchstg].
- Complexity O(max(P,R)·M·d) with P, R ≪ N. Up to **10× training speedup** and **4× memory reduction** on 8,600-node CA dataset[^src-patchstg].

## Performance

SOTA on all four LargeST benchmarks (SD 716, GBA 2,352, GLA 3,834, CA 8,600 nodes) vs 10 baselines. CA dataset (largest): MAE 17.35 (avg), MAPE 12.79%. Ablation: leaf KDTree is the single most critical component[^src-patchstg].

## Design Philosophy

PatchSTG occupies a unique position among efficient dynamic spatial modeling paradigms[^src-patchstg]:

| Paradigm | Complexity | Information Loss | Interpretability | Domain Knowledge |
|----------|-----------|-------------------|------------------|------------------|
| Dot-Product (D2STGNN) | O(N²d) | ✗ | ✓ | ✗ |
| Linear (BigST) | O(Nd²) | ✗ | ✗ | ✗ |
| Low-Rank (STWave) | O(NRd) | ✓ | ✗ | ✗ |
| **Patching (PatchSTG)** | O(NRd) | ✗ | ✓ | ✓ |

PatchSTG is the only paradigm that simultaneously avoids information loss, maintains interpretability, and incorporates domain knowledge (spatial locality).

## Context

- Related to [[traffic-forecasting]], [[large-scale-spatial-temporal-graph]], and [[ragc|RAGC]] (another large-scale efficiency method).
- Contrasts with low-rank methods like [[specstg|SpecSTG]] and [[ustd|USTD]] that sacrifice fidelity for speed.
- The patching paradigm parallels vision models: [[source-patchtst|PatchTST]] (temporal patching) and ViT (spatial patching).
- [[stunet|STUNet]] (KDD 2026) also uses patching, but on the **adjacency matrix** (not geo-points) to build frozen spatial tokens for **cross-network zero-shot**; PatchSTG is the main in-domain large-scale baseline STUNet compares against on LargeST[^src-stunet].
- [[lets-group|Let's Group]] (IJCAI 2025, whose reference cites this work's arXiv version [Fang et al., 2024, arXiv:2412.09972]) classifies PatchSTG's geographic-coordinate partitioning — together with FCGCN (Louvain) and LarSTL (METIS) — as **static subgraph partitioning** that fails to capture dynamic spatio-temporal dependencies, and proposes partitioning by feature similarity with learnable memory vectors instead (Sec. 1–2 of that paper, 作者自述)[^src-lets-group].

[^src-patchstg]: [[source-patchstg]]
[^src-stunet]: [[source-stunet]]
[^src-lets-group]: [[source-lets-group]]
