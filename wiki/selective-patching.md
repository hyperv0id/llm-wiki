---
title: "Selective Patching"
type: technique
tags:
  - time-series
  - forecasting
  - patch
  - differentiable-selection
  - neurips-2025
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: high
status: active
---

# Selective Patching

**Selective Patching** is the first operator in the [[selective-representation-space|SRS]] module (Wu et al., NeurIPS 2025). It replaces fixed-stride adjacent patch retrieval with **adaptive, sample-wise selection** of $n$ patches of size $p$ from all stride-1 candidates in a padded look-back window.[^src-srsnet]

## Method

1. Scan padded context $X'$ with stride 1 to form $K=(n-1)\cdot s+1$ candidate patches $P'\in\mathbb{R}^{N\times K\times p}$.[^src-srsnet]
2. MLP **Scorer$_s$**: $P'\mapsto S^{s}\in\mathbb{R}^{N\times K\times n}$ — $n$ scores per patch to allow sampling **with replacement**.[^src-srsnet]
3. $I^{s}=\mathrm{Argmax}(S^{s})$ selects indices; selected patches are reweighted by a detach-reciprocal Hadamard product so hard Argmax remains exact while gradients flow through scores.[^src-srsnet]

Repeated selection lets beneficial patches appear multiple times, expanding the combinatorial representation space to $C_{K+n-1}^{n}$ multisets before ordering.[^src-srsnet]

## Role in SRS

Selective Patching **decides which subsequences** enter the representation space and has the largest ablation impact among SRS components (vs Dynamic Reassembly and Adaptive Fusion).[^src-srsnet] Selected patches then enter [[dynamic-reassembly|Dynamic Reassembly]] for ordering and [[adaptive-fusion|Adaptive Fusion]] with conventional embeddings.

## Connections

- Parent concept: [[selective-representation-space]]
- Downstream: [[dynamic-reassembly]], [[adaptive-fusion]], [[srsnet]]
- Contrast: [[patch-based-tokenization]] (fixed adjacent patches)
- Source: [[source-srsnet]]

---

[^src-srsnet]: [[source-srsnet]]
