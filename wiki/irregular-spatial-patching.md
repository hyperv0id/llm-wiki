---
title: "Irregular Spatial Patching"
type: technique
tags:
  - spatial-temporal
  - kdtree
  - spatial-partitioning
created: 2026-06-08
last_updated: 2026-08-31
source_count: 1
confidence: high
status: active
---

# Irregular Spatial Patching

Irregular Spatial Patching is the core spatial data management technique introduced in [[patchstg|PatchSTG]] (KDD 2025) to enable efficient Transformer-based traffic forecasting on **irregularly distributed** spatial points[^src-patchstg]. It solves the fundamental mismatch between vision Transformer patching (regular grid pixels → equal-size patches) and traffic data (irregular sensor locations).

## Motivation

Dynamic spatial attention in traffic models has quadratic complexity O(N²d), making it intractable for thousands of sensors. Vision Transformers reduce this by patching neighboring pixels together. But traffic sensors are irregularly placed on roads — they cannot simply be divided into equal-size rectangular patches[^src-patchstg].

## Three-Stage Pipeline

### 1. Leaf KDTree Partitioning

A novel [[leaf-kdtree|leaf KDTree]] recursively partitions traffic points by latitude/longitude into balanced leaf nodes of capacity C (typically 2-3 points). Unlike standard KDTree, leaf KDTree ensures **all points are stored in leaf nodes**, avoiding hyperplane points being stranded in internal nodes[^src-patchstg].

BFS traversal yields new indices where leaf nodes from the same subtree are adjacent, ensuring spatial locality.

### 2. Cosine-Similarity Padding

N is rarely divisible by C, leaving unfull leaf nodes. PatchSTG pads unfull nodes with the **most temporally similar** points from other leaf nodes (via cosine similarity on traffic time series), rather than zeros or nearest neighbors[^src-patchstg]. This ensures non-overlap: a point may appear in multiple patches, but not multiple times within the same patch.

### 3. Subtree Backtracking Patching

Leaf nodes from the same subtree maintain strong spatial correlations. PatchSTG backtracks from leaf nodes up the tree to merge subtrees into patches of size P = C × Nₚ, where Nₚ must be a power of 2 (binary tree constraint). This mitigates the **unbalanced padding issue** — unfull leaf nodes are padded by points similar to *different* points in the same subtree[^src-patchstg].

## Properties

| Property | Mechanism |
|----------|-----------|
| **Balanced** | Leaf KDTree + subtree backtracking ensures equal-occupancy patches |
| **Non-overlapping** | Cosine-similarity padding from *other* leaf nodes avoids self-repetition |
| **Interpretable** | Spatial locality preserved; patches visually correspond to adjacent geographic regions |
| **Efficient** | O(N log N) preprocessing; does not affect training loop complexity |

## Ablation Evidence

- Removing leaf KDTree ("w/o LKDT") causes the largest performance drop — spatial message passing on irrelevant (non-adjacent) points is worse than no spatial modeling at all[^src-patchstg].
- METIS (balanced graph partition) fails because its recursive merge semantics differ and cannot produce the balanced padded non-overlap patches needed for dual attention[^src-patchstg].
- KMeans produces highly unbalanced patches (max/min ratio up to 96:6 on SD dataset)[^src-patchstg].

## Applications

Currently demonstrated in PatchSTG for traffic forecasting. The paper suggests extension to other irregular spatial tasks like **national air quality prediction**[^src-patchstg].

Patch 划分产出的 $R\times P$ 网格直接决定后续 dual attention 的稀疏掩码结构（块对角 × 位置对齐），形式化见 [[patchstg-sparse-attention-form]]。

[^src-patchstg]: [[source-patchstg]]
