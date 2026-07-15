---
title: "SRSNet: Enhancing Time Series Forecasting through Selective Representation Spaces"
type: source-summary
tags:
  - time-series-forecasting
  - patch
  - selective-representation-space
  - plug-and-play
  - long-term-forecasting
  - neurips-2025
created: 2026-07-13
last_updated: 2026-07-21
source_count: 0
confidence: low
status: active
---

## Summary

**SRSNet / SRS** (Selective Representation Space) is a NeurIPS 2025 paper by Xingjian Wu, Xiangfei Qiu, Hanyin Cheng, Zhengyu Li, Jilin Hu, Chenjuan Guo, and Bin Yang (ECNU; arXiv:2510.14510). It argues that conventional adjacent patching builds a **fixed** representation space and can break periods or admit anomalies and distributional shifts. The proposed modular [[selective-representation-space|SRS]] module instead adaptively selects and reorders patches, then fuses them with conventional patch embeddings as a plug-and-play front-end for patch-based forecasters. [[srsnet|SRSNet]] is the simple instantiation SRS + MLP head and reports multi-domain LTSF SOTA.

## Core Arguments

**1. Fixed adjacent patching is brittle.** Fixed-stride patches assume useful forecast information is evenly distributed. Changeable periods, anomalies, and shifting violate that assumption, so adjacent patches may lose periodic semantics or inject noise into the representation space.

**2. Selective Representation Space.** SRS scans all stride-1 candidate patches, uses MLP Scorer$_s$ for [[selective-patching|Selective Patching]] (with replacement), then Scorer$_r$ for [[dynamic-reassembly|Dynamic Reassembly]] (learned order). Differentiable hard selection/sort reuses scores with a detach-reciprocal Hadamard trick so Argmax/Argsort stay exact while gradients flow. Search space size is $C_{K+n-1}^{n}\cdot n!$.

**3. Adaptive Fusion + CI pipeline.** Instance-normalized series are channel-independent. [[adaptive-fusion|Adaptive Fusion]] forms the convex combination $\tilde{E}=\alpha\odot E^{c}+(1-\alpha)\odot E^{s}$ between conventional and selective patch embeddings, then adds positional embeddings for any unchanged patch backbone.

**4. Simple backbone is enough.** SRSNet flattens SRS embeddings through an MLP (≤2 layers) trained with MSE, arguing a better representation space can outweigh complex inductive bias in mediocre spaces.

## Experiments

Eight benchmarks (ETT×4, Weather, Electricity, Solar, Traffic); horizons $\{96,192,336,720\}$; look-back in $\{96,336,512\}$. Baselines include TimeKAN, Amplifier, iTransformer, TimeMixer, PatchTST, Crossformer, TimesNet, DLinear, Stationary, FEDformer. SRSNet leads most average rankings (e.g., ETTh1 MSE 0.404, Solar 0.183, Traffic 0.392). As a plugin, SRS improves MLP ~5–16% and PatchTST / Crossformer / PatchMLP / xPatch by roughly ~5% on average; Selective Patching has the largest ablation impact. Efficiency: lighter than many Transformers; ~10% memory/time and <5% MACs overhead when plugged into PatchTST/Crossformer.

## Limitations

Impractical for non-patch models (patch size 1 loses patch semantics). Scaling-law behavior under billion-scale foundation pretraining is unverified. Selected patches are useful for forecasting but not guaranteed interpretable. Fusion weights $\alpha$ benefit from dataset-aware initialization (higher for periodic/stationary series). Future work: environment-aware patch patterns, MoE routers for heterogeneous corpora, better $\alpha$ supervision.

## Key Terminology

- **Selective Representation Space (SRS)**: adaptive patch selection + reordering + fusion front-end
- **Selective Patching / Dynamic Reassembly**: score-and-select / score-and-sort patch operators
- **SRSNet**: SRS + MLP forecasting head

---

