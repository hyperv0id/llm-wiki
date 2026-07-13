---
title: "SRSNet"
type: entity
tags:
  - time-series
  - forecasting
  - patch
  - mlp
  - selective-representation-space
  - neurips-2025
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: high
status: active
---

# SRSNet

**SRSNet** is a long-term multivariate time series forecasting model that pairs the modular [[selective-representation-space|Selective Representation Space (SRS)]] front-end with a lightweight MLP head. Proposed by Wu et al. (ECNU) at NeurIPS 2025, it is designed to show that adaptively constructed patch representations can outperform complex fixed-patch backbones on multi-domain LTSF benchmarks.[^src-srsnet]

## Architecture

Pipeline (channel-independent after instance normalization):[^src-srsnet]

1. **Conventional adjacent patching** — fixed-stride baseline patches $P$
2. **[[selective-patching|Selective Patching]]** — score all stride-1 candidates; pick $n$ patches (with replacement)
3. **[[dynamic-reassembly|Dynamic Reassembly]]** — learn a permutation of the selected patches
4. **[[adaptive-fusion|Adaptive Fusion]]** — convex combination of conventional vs selective patch embeddings + position embedding
5. **MLP head** — flatten embeddings $\rightarrow$ forecast horizon $L$; train with MSE

The paper emphasizes universal approximation: with a better representation space, a shallow MLP (≤2 layers) is theoretically and empirically competitive.[^src-srsnet]

## Results (highlights)

On eight public datasets with horizons $\{96,192,336,720\}$, SRSNet attains many first-place average rankings among recent Transformer/CNN/KAN/MLP/Linear baselines, e.g. ETTh1 MSE **0.404**, ETTh2 **0.334**, Solar **0.183**, Traffic **0.392**.[^src-srsnet]

As a **plugin**, SRS also improves PatchTST, Crossformer, PatchMLP, and xPatch (≈5% average) and naive MLP by ~5–16% depending on dataset—largest relative gains when the backbone is simple.[^src-srsnet]

## Efficiency

SRSNet is lighter than multi-layer Transformers such as Crossformer/FEDformer/PatchTST on reported ETTh1/Solar settings, and balances accuracy vs DLinear/Amplifier which are cheaper but weaker on large Solar/Traffic-scale series. Plugging SRS into PatchTST/Crossformer adds roughly ~10% memory/time and <5% MACs.[^src-srsnet]

## Connections

- **Core module**: [[selective-representation-space]], [[selective-patching]], [[dynamic-reassembly]], [[adaptive-fusion]]
- **Patch lineage**: [[patch-based-tokenization]], [[patchtst]], [[crossformer]], [[channel-independence]], [[instance-normalization]]
- **Related LTSF baselines**: [[itransformer]], [[timemixer]], [[timesnet]], [[ltsf-linear]], [[fedformer]]
- **Source**: [[source-srsnet]]

---

[^src-srsnet]: [[source-srsnet]]
