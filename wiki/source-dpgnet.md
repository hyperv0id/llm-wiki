---
title: "DPGNet: Modeling Dynamic Graphs and Complex Temporal Patterns for Spatiotemporal Forecasting"
type: source-summary
tags:
  - spatiotemporal-forecasting
  - dynamic-graph-learning
  - traffic-forecasting
  - temporal-decomposition
  - multi-scale-learning
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Source: DPGNet (ICLR 2026, under review)

**DPGNet** (Dynamic Graph Prediction Network) is a spatiotemporal forecasting model submitted to ICLR 2026 under double-blind review[^src-dpgnet]. It addresses two limitations of existing models: inability to capture dynamic inter-node relationships, and inability to comprehensively model complex temporal patterns. The paper proposes two core components: the **Adaptive Graph Learner (AGL)** and the **Adaptive Season Learner (ASL)**.

## Core Contributions

1. **AGL** — A plug-and-play graph structure generator that captures dynamic implicit relationships between nodes while suppressing weak connections[^src-dpgnet]. Composed of L stacked G-RNN units, each integrating self-attention with a gating mechanism: the update gate incorporates implicit relationships from current data into the adjacency matrix, while the reset gate discards noise and weak connections. Uses patch processing to reduce G-RNN steps from L to b ≪ L, mitigating RNN error accumulation. The hidden state C_t tracks adjacency matrix evolution across time steps, enhancing interpretability[^src-dpgnet].

2. **ASL** — A spatiotemporal pattern learning module combining temporal decomposition (moving average → trend + seasonal), multi-scale processing (average pooling downsample to m scales), and pattern- and scale-specific graph construction[^src-dpgnet]. Trend features extracted via two-layer dilated TCN; seasonal features via FFT→Linear→iFFT frequency-domain approach. Constructs separate adjacency matrices Ã_T (trend) and Ã_S (seasonal) per scale, then fuses via bottom-up (seasonal, fine→coarse) and top-down (trend, coarse→fine) strategies[^src-dpgnet].

3. **Experimental results** — DPGNet outperforms five recent baselines (PGCN, PMC-GCN, STIDGCN, TESTAM, WAVGCRN) on five datasets: METR-LA, PEMS-Bay, PEMS08, Electricity, Weather. AGL replacement experiments show 85% improvement rate across 5 base models. DPGNet achieves SOTA with only 184K trainable parameters, demonstrating favorable efficiency-accuracy balance[^src-dpgnet].

## Limitations & Caveats

- ⚠️ Under review at ICLR 2026 — findings are preliminary and not yet peer-reviewed
- No automatic selection mechanism for multi-scale hyperparameter m in ASL
- Requires predefined graph (adjacency matrix A) — cannot yet handle graph-free scenarios
- Single paper source — confidence: medium

[^src-dpgnet]: [[source-dpgnet]]
