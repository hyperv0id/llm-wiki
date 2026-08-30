---
title: "HEPHAESTUS"
type: entity
tags:
  - traffic-forecasting
  - spatio-temporal
  - mixture-of-experts
  - multi-scale
  - iclr-2026
created: 2026-06-08
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# HEPHAESTUS (Hierarchical Periodic Heterogeneous Adaptive Spatio-Temporal Unified System)

HEPHAESTUS is a unified spatio-temporal traffic forecasting framework proposed by anonymous authors, under review at ICLR 2026. The name is an acronym referencing the Greek god of craftsmanship, reflecting the model's adaptive routing and heterogeneous fusion mechanisms[^src-hephestus].

## Architecture

HEPHAESTUS processes input traffic sequences through a pipeline of five stages[^src-hephestus]:

1. **RevIN normalization** — instance normalization for distribution shift handling.
2. **AMS-MoE** — Adaptive Multi-Scale Mixture of Experts with Moving-Patch and top-K routing.
3. **ST-Blocks** — Stacked blocks each containing PTA (temporal) followed by HSA (spatial).
4. **Multi-Scale Aggregator** — Weighted fusion of expert outputs via learned routing weights.
5. **MLP Predictor + Inverse RevIN** — Final forecasting and denormalization.

## Key Components

### [[ams-moe|AMS-MoE]] (Adaptive Multi-Scale Mixture of Experts)

Dynamic routing across M patch-scale experts. Each expert is a Transformer encoder at a fixed patch size. A temporal decomposition-based router extracts features from the raw input, computes soft assignment probabilities with Gaussian noise injection, and applies Top-K sparsification. Optimal: M=4 experts, K=2[^src-hephestus].

### [[periodic-temporal-attention|PTA]] (Periodic Temporal Attention)

Explicit daily/weekly periodicity modeling via learnable embedding matrices PD (288 daily intervals) and PW (2016 weekly intervals). These are projected to query space, broadcast across spatial nodes, and used for cross-attention with standard key/value projections[^src-hephestus].

### [[heterogeneous-spatial-attention|HSA]] (Heterogeneous Spatial Attention)

Balances global shared patterns and node-specific behaviors. Uses node embeddings Se and a low-rank pattern library PL (rank r=8 optimal). A Common Linear provides shared value projection; a Specific Linear decomposes per-node transformations via tensor product of node embeddings and pattern library. Gated fusion dynamically mixes both[^src-hephestus].

## Performance

HEPHAESTUS achieves SOTA across all 6 traffic benchmarks vs 15 baselines[^src-hephestus]:

| Dataset | MAE vs best baseline | Key baseline beaten |
|---------|---------------------|---------------------|
| METR-LA | 2.62 (H=12) vs DGCRN 3.42 | All STGNNs |
| PEMS-BAY | 1.88 (H=12) vs PathFormer 2.76 | All STGNNs |
| PEMS03 | 14.76 vs DGCRN 14.63 | ~tied |
| PEMS04 | 18.21 vs ASTGCN 20.15 | Significant gap |
| PEMS07 | 19.18 vs STID 19.61 | Clear win |
| PEMS08 | 13.56 vs STID 14.21 | Clear win |

## Efficiency

716K parameters, 5475MB GPU memory during training on PEMS04. Training: 62.07s/epoch; inference: 6.49s/epoch. Lower memory than [[pdformer|PDFormer]] (8295MB) and PathFormer (6652MB)[^src-hephestus].

## Case Study: Dynamic Scale Selection

Visualization of routing behavior shows: during peak hours (rapid traffic changes), the router selects experts with smaller patch sizes for fine-grained capture; during off-peak (smooth patterns), it shifts to larger patch sizes for long-term trend modeling[^src-hephestus].

## Relationship to Other Models

| Model | Relationship |
|-------|-------------|
| [[timemixer|TimeMixer]] | Both perform multi-scale mixing. TimeMixer uses fixed down-sampling + seasonal/trend mixing; HEPHAESTUS replaces fixed decomposition with input-adaptive MoE routing. HEPHAESTUS cites TimeMixer (Wang et al., 2024b;a) in related work[^src-hephestus]. |
| [[phat|PHAT]] | Both address temporal heterogeneity and are at ICLR 2026. PHAT uses FFT-based period detection + bucketing + PNA attention; HEPHAESTUS uses AMS-MoE dynamic routing with periodic attention. Complements rather than competes — PHAT excels at cross-variate period alignment, HEPHAESTUS at input-adaptive scale selection[^src-hephestus]. |
| [[patchtst|PatchTST]] | HEPHAESTUS's Moving-Patch extends PatchTST's patching with boundary replication and overlapping stride-1 extraction for dense temporal coverage; PatchTST lacks spatial awareness[^src-hephestus]. |
| [[pathformer|PathFormer]] | Both use multi-scale patching (PathFormer: adaptive pathways). HEPHAESTUS uses MoE routing instead of learnable pathways[^src-hephestus]. |
| [[gwnet|GWNet]] | GWNet's adaptive adjacency matrix is a precursor to HSA's node-specific spatial modeling; HSA adds gated fusion with shared global patterns[^src-hephestus]. |

## Caveats

- ⚠️ Under review at ICLR 2026 — not yet accepted. All results preliminary.
- Single-source: only one paper supports these findings.
- No code released yet.
- Traffic-only evaluation — cross-domain generalization untested.

[^src-hephestus]: [[source-hephestus]]
