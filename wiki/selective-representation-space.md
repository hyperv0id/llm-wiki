---
title: "Selective Representation Space (SRS)"
type: concept
tags:
  - time-series
  - forecasting
  - patch
  - representation-learning
  - plug-and-play
  - neurips-2025
created: 2026-07-13
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

# Selective Representation Space (SRS)

**Selective Representation Space (SRS)** is a plug-and-play module for patch-based time series forecasting that **adaptively constructs** the patch representation space of a look-back window, instead of using fixed adjacent patches only. Introduced by Wu et al. (NeurIPS 2025) with the simple instantiation [[srsnet|SRSNet]].[^src-srsnet]

## Motivation

[[patch-based-tokenization|Adjacent patching]] (fixed stride) creates a **fixed** representation space across contexts. Under changeable periods, anomalies, and shifting, that space can break period semantics or include harmful noise. Multi-scale patch sizes enlarge the set of spaces but still use fixed strides.[^src-srsnet]

SRS reframes patching as selecting informative subsequences and ordering them for permutation-sensitive backbones, optimizing selection end-to-end for forecast loss.[^src-srsnet]

## Components

| Component | Role |
|-----------|------|
| [[selective-patching\|Selective Patching]] | From $K$ stride-1 candidates, pick $n$ patches (with replacement) via Scorer$_s$ |
| [[dynamic-reassembly\|Dynamic Reassembly]] | Learn order of selected patches via Scorer$_r$ + Argsort |
| [[adaptive-fusion\|Adaptive Fusion]] | Convex mix of conventional vs selective patch embeddings with $\alpha$ |

Together they explore a combinatorial space of size $C_{K+n-1}^{n}\cdot n!$ with gradient-based search, then feed any unchanged patch encoder/decoder.[^src-srsnet]

## Empirical claims

- Standalone [[srsnet|SRSNet]] (SRS+MLP) is multi-domain LTSF competitive / SOTA vs recent baselines.[^src-srsnet]
- As a plugin, improves PatchTST, Crossformer, PatchMLP, xPatch by ~5% on average; Selective Patching dominates ablations.[^src-srsnet]
- Orthogonal to multi-scale patching: can be applied per scale in PatchMLP-style models.[^src-srsnet]
- Modest overhead when plugged into classical patch Transformers (~10% time/memory, <5% MACs).[^src-srsnet]

## Limitations

Not natural for non-patch models; foundation-model scale behavior unproven; limited interpretability of selected patches; $\alpha$ initialization benefits from stationarity/periodicity priors.[^src-srsnet]

## Related

- [[patch-based-tokenization]] — fixed adjacent baseline that SRS relaxes
- [[patchtst]], [[crossformer]] — canonical patch backbones enhanced by SRS
- [[channel-independence]], [[instance-normalization]] — default preprocessing stack in the paper
- [[source-srsnet]]

---

[^src-srsnet]: [[source-srsnet]]
