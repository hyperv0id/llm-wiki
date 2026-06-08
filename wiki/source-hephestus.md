---
title: "Source: HEPHAESTUS"
type: source-summary
tags:
  - traffic-forecasting
  - spatio-temporal
  - multi-scale-modeling
  - mixture-of-experts
  - iclr-2026
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Source: HEPHAESTUS — ICLR 2026 (Under Review)

**Full Title**: HEPHAESTUS: Hierarchical Periodic Heterogeneous Adaptive Spatio-Temporal Unified System for Traffic Forecasting

**Authors**: Anonymous (double-blind review)

**Venue**: ICLR 2026, under review

**Link**: `raw/hephestus-iclr2026.pdf`

## Core Contribution

HEPHAESTUS is a unified spatio-temporal traffic forecasting framework integrating three innovations[^src-hephestus]:

1. **AMS-MoE (Adaptive Multi-Scale Mixture of Experts)**: Input-adaptive dynamic routing across multiple patch-scale experts via Moving-Patch (boundary replication + overlapping stride-1 extraction + linear projection) and Top-K sparse gating, replacing fixed-scale decomposition.
2. **PTA (Periodic Temporal Attention)**: Time-aware cross-attention with learnable daily (288 intervals) and weekly (288×7 intervals) periodic embedding matrices as query, broadcasting across spatial nodes.
3. **HSA (Heterogeneous Spatial Attention)**: Node embeddings Se combined with low-rank pattern library PL via tensor decomposition to generate per-node adaptive value projections, fused with shared global patterns via sigmoid-gated mixing.

## Key Results

- SOTA across 6 benchmarks (METR-LA, PEMS-BAY, PEMS03/04/07/08) vs 15 baselines including STGCN, DCRNN, GWNet, ASTGCN, MTGNN, DGCRN, PatchTST, PathFormer, iTransformer[^src-hephestus].
- 716K parameters; 5475MB GPU memory (lower than PDFormer's 8295MB, PathFormer's 6652MB)[^src-hephestus].
- Ablation: AMS-MoE removal causes largest degradation; PTA and HSA both independently critical[^src-hephestus].
- Optimal config: M=4 experts, K=2 sparsity, r=8 pattern library rank[^src-hephestus].

## Limitations and Caveats

- **Under review** — not yet accepted at ICLR 2026. All findings are preliminary.
- Single-source confidence assessment; results await peer review.
- No code available until acceptance.
- Evaluated only on traffic datasets; generalizability to other time series domains unverified.

[^src-hephestus]: [[source-hephestus]]
