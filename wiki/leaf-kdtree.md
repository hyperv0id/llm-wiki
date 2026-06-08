---
title: "Leaf KDTree"
type: technique
tags:
  - spatial-partitioning
  - kdtree
  - tree-algorithm
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# Leaf KDTree

Leaf KDTree is a variant of the classical KDTree (K-dimensional tree) introduced in [[patchstg|PatchSTG]] (KDD 2025) to enable balanced, non-overlapping spatial partitioning of irregularly distributed traffic points[^src-patchstg]. The key difference from standard KDTree: **all points are guaranteed to reside in leaf nodes**, eliminating the issue of hyperplane points being "lost" in internal nodes.

## Motivation

Standard KDTree uses internal nodes as partitioning hyperplanes (the median point of the selected axis), which means those hyperplane points are **not** placed into leaf nodes. During BFS traversal, this causes points that are spatially distant to appear adjacent in the traversal order — breaking the spatial locality needed for efficient patching[^src-patchstg].

## Algorithm

Leaf KDTree maintains the binary tree structure of standard KDTree but modifies the hyperplane rule[^src-patchstg]:

- **Even number of points**: Use the median value as the hyperplane, splitting points between the two children. The median point is assigned to one child.
- **Odd number of points**: Use the value *between* the median point and its left neighbor as the hyperplane. This pushes all points — including the median — into leaf children.

The tree is built by recursing on each child until leaf nodes reach capacity C (typically 2-3 points). Splitting axes alternate between latitude and longitude.

After construction, **Breadth-First Search (BFS)** on the tree produces a new ordering where leaf nodes from the same subtree are adjacent — critical for the subsequent [[irregular-spatial-patching|patching step]][^src-patchstg].

## Design Rationale

| Standard KDTree | Leaf KDTree |
|----------------|-------------|
| Internal nodes contain hyperplane points | All points in leaf nodes |
| BFS order may mix distant points | BFS order preserves spatial locality |
| Suitable for nearest-neighbor search | Suitable for **spatial partitioning** and patching |

## Role in PatchSTG

Leaf KDTree is the **first stage** of PatchSTG's [[irregular-spatial-patching]] pipeline[^src-patchstg]:

1. Leaf KDTree (latitude, longitude, capacity C) → balanced leaf nodes
2. BFS → reordered indices preserving subtree adjacency
3. Cosine-similarity padding → fill unfull leaf nodes
4. Subtree backtracking → merge leaf nodes into equal-size patches

It is the most critical component of PatchSTG: ablation "w/o LKDT" (running dual attention directly on original input) causes severe performance degradation, confirming that spatial message passing should only happen between geographically adjacent points[^src-patchstg].

## Complexity

O(N log N) for tree construction, done entirely in pre-processing and not in the training loop[^src-patchstg].

## Significance

Leaf KDTree bridges **spatial data management** (databases, GIS) with **deep learning** by providing a principled, interpretable way to partition irregular spatial data for Transformer attention. It enables the patching paradigm — successful in vision — to transfer to irregular spatial domains for the first time[^src-patchstg].

[^src-patchstg]: [[source-patchstg]]
