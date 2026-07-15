---
title: "Dynamic Reassembly"
type: technique
tags:
  - time-series
  - forecasting
  - patch
  - permutation
  - differentiable-sorting
  - neurips-2025
created: 2026-07-13
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

# Dynamic Reassembly

**Dynamic Reassembly** is the second operator in the [[selective-representation-space|SRS]] module (Wu et al., NeurIPS 2025). After [[selective-patching|Selective Patching]] chooses $n$ patches, Dynamic Reassembly **learns their order**, because most patch-based backbones are permutation-variant and sensitive to sequence structure.[^src-srsnet]

## Method

1. MLP **Scorer$_r$** maps selected patches to a single score per patch: $S^{r}\in\mathbb{R}^{N\times n}$.[^src-srsnet]
2. $I^{r}=\mathrm{Argsort}(S^{r})$ yields a ranking; patches are reordered accordingly.[^src-srsnet]
3. Differentiable hard sort uses the same detach-reciprocal Hadamard trick as Selective Patching so Argsort stays exact while gradients attach to scores.[^src-srsnet]

This multiplies the search space by $n!$ relative to selection alone, jointly optimized with forecasting loss via gradient descent.[^src-srsnet]

## Empirical role

Ablations show Dynamic Reassembly improves over selection-only variants, but typically less than Selective Patching; it supplies additional candidate representation spaces by reordering useful patches for subsequent encoders.[^src-srsnet]

## Connections

- Upstream: [[selective-patching]]
- Parent: [[selective-representation-space]]
- Downstream: [[adaptive-fusion]], [[srsnet]]
- Source: [[source-srsnet]]

---

[^src-srsnet]: [[source-srsnet]]
