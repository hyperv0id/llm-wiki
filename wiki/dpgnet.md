---
title: "DPGNet"
type: entity
tags:
  - spatiotemporal-forecasting
  - traffic-forecasting
  - dynamic-graph-learning
  - temporal-decomposition
  - multi-scale-learning
  - graph-neural-networks
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# DPGNet — Dynamic Graph Prediction Network

**DPGNet** (Dynamic Graph Prediction Network) is a spatiotemporal forecasting model, under review at ICLR 2026, that addresses two key weaknesses of existing STG models: inability to capture **dynamic implicit relationships** between nodes and inability to comprehensively model **complex temporal patterns**[^src-dpgnet]. It introduces two core components: the [[adaptive-graph-learner|AGL]] and the [[adaptive-season-learner|ASL]].

## Architecture Overview

Given input $X \in \mathbb{R}^{N \times L \times C}$ (N nodes, L sequence length, C features) and a predefined graph $A$, DPGNet predicts future sequence $\hat{X} \in \mathbb{R}^{N \times O \times C}$[^src-dpgnet]:

```
X → Spatial-Temporal Embedding → AGL → ASL → Output Layer → X̂
```

### 1. Spatial-Temporal Embedding Layer

Normalizes input along both temporal and spatial dimensions, then embeds via linear layers. Also incorporates timestamp features $S \in \mathbb{R}^{L \times M}$[^src-dpgnet]:

$$H = \text{Linear}(\text{Norm}_T(X)) + \text{Linear}(\text{Norm}_S(X)) + \text{Linear}(S)$$

### 2. [[adaptive-graph-learner|AGL — Adaptive Graph Learner]]

A plug-and-play module of L stacked G-RNN units. Each G-RNN integrates self-attention with gating[^src-dpgnet]:

- **Update gate** $i$: Incorporates implicit relationships $A_t$ into the adjacency matrix
- **Reset gate** $f = 1 - i$: Discards noise and weak connections
- **Hidden state** $C_t$: Tracks adjacency matrix evolution, initialized as $C_0 = A$ (explicit graph)
- **Output** $A_L$: Guides spatial aggregation in subsequent modules

Uses patch processing to compress L time steps into b patches (b ≪ L), reducing G-RNN steps and alleviating error accumulation[^src-dpgnet].

### 3. [[adaptive-season-learner|ASL — Adaptive Season Learner]]

Combines three strategies for temporal pattern modeling[^src-dpgnet]:

1. **Multi-scale processing**: Average pooling downsamples H to m scales ($H^0$…$H^m$), from finest to coarsest
2. **Temporal decomposition**: Moving average extracts trend $H^i_T$ and seasonal $H^i_S = H^i - H^i_T$
3. **Pattern-specific graph construction**: Separate adjacency matrices $\tilde{A}^i_T$ (trend) and $\tilde{A}^i_S$ (seasonal) per scale

Trend features via TCN (two stacked dilated convolutions); seasonal features via FFT → Linear → iFFT[^src-dpgnet]. Multi-scale fusion: bottom-up for seasonal (fine→coarse), top-down for trend (coarse→fine)[^src-dpgnet].

### 4. Output Layer

Predictions from all scales aggregated: $\hat{X}^i = \text{FFD}_i(Z^i)$, $\hat{X} = \sum_{i=0}^m \hat{X}^i$[^src-dpgnet].

## Performance

### Regular-term forecasting (12→3/6/12 steps)

DPGNet outperforms all five baselines (PGCN, PMC-GCN, STIDGCN, TESTAM, WAVGCRN) on the majority of scenarios across METR-LA, PEMS-Bay, PEMS08, Electricity, and Weather datasets[^src-dpgnet]. On METR-LA, average MAE reduction of 3.65% vs second-best across all horizons.

### Long-term forecasting (12→24/36/48 steps)

Strong advantage, especially at the longest horizons[^src-dpgnet]:
- METR-LA: MAE reduction 7.42% at 240 min vs 3.34% at 120 min
- PEMS08: MSE reduction averaging 9.31%

### Efficiency

- **184K trainable parameters** — lowest among all compared baselines[^src-dpgnet]
- Competitive inference time (behind only STIDGCN and TESTAM)

## AGL as Plug-and-Play

AGL replacement experiments show that swapping baseline graph generators (GWNet, PMC-GCN, STIDGCN, STGCN, WAVGCRN) with AGL improves accuracy in 85% of scenarios, while introducing fewer or comparable trainable parameters[^src-dpgnet]. For models using only predefined adjacency (STGCN, WAVGCRN), AGL adds negligible parameters.

## Ablation

All core modules are essential[^src-dpgnet]:
- Removing AGL (replacing $A_L$ with $A$): MAE ↑12.70%, MSE ↑9.76% at 240 min
- Removing seasonal features: ↑MAE and MSE at all horizons
- Removing trend features: ↑MAE and MSE (less severe than removing seasonal)

## Related Pages

- [[source-dpgnet]] — Source summary
- [[adaptive-graph-learner]] — AGL technique in detail
- [[adaptive-season-learner]] — ASL technique in detail
- [[gwnet]] — GWNet, whose adaptive adjacency is directly replaced/improved by AGL
- [[dcrnn]] — DCRNN, the diffusion convolution predecessor
- [[traffic-forecasting]] — Traffic forecasting overview
- [[spatio-temporal-decomposition]] — Decomposition paradigm in STG models
- [[multi-scale-linear-prediction]] — DST-Mamba's multi-scale trend modeling
- [[timemixer]] — TimeMixer, the multi-scale mixing predecessor
- [[autoformer]] — Autoformer, the original temporal decomposition transformer

[^src-dpgnet]: [[source-dpgnet]]
