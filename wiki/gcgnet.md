---
title: "GCGNet"
type: entity
tags:
  - time-series-forecasting
  - exogenous
  - graph-neural-network
  - variational-autoencoder
  - generative-model
  - iclr-2026
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: high
status: active
---

# GCGNet

**GCGNet** (Graph-Consistent Generative Network) is an ECNU Decision Intelligence model for **time series forecasting with historical and (optional) future exogenous variables**, published at ICLR 2026 (arXiv:2603.08032). Code: https://github.com/decisionintelligence/GCGNet.[^src-gcgnet]

## Problem

Predict future endogenous series \(Y^{\mathrm{endo}}\) from historical endogenous \(X^{\mathrm{endo}}\), historical exogenous \(X^{\mathrm{exo}}\), and optionally future exogenous \(Y^{\mathrm{exo}}\). The paper argues that most deep exogenous forecasters use a **two-step** strategy (temporal→channel or channel→temporal), which interferes with learning [[joint-temporal-channel-correlation|joint temporal and channel correlations]], and that raw-observation correlation learning is brittle under noise.[^src-gcgnet]

## Architecture

Three modules (Figure 3 of the paper):[^src-gcgnet]

| Module | Role |
|--------|------|
| [[variational-generator-exogenous\|Variational Generator]] | Instance norm + VAE coarse forecast \(\tilde Y^{\mathrm{endo}}\) (and \(\tilde Y^{\mathrm{exo}}\) if future exo unavailable) |
| [[graph-structure-aligner\|Graph Structure Aligner]] | Patchify full sequences; Graph Learner + Graph VAE → adjacency; align \(A\) vs \(\hat A\) with \(L_1\) |
| [[graph-refiner\|Graph Refiner]] | Top-k sparsify \(\hat A\); multi-layer GCN on patch nodes; flatten + linear head → \(\hat Y^{\mathrm{endo}}\) |

Training objective: forecasting L1 + graph alignment + two KL regularizers (generator VAE and Graph VAE).[^src-gcgnet]

## Empirical Snapshot

On 12 real exogenous datasets (EPF markets, Chilean energy/hydro, Longyuan wind + ERA5), GCGNet leads most MSE/MAE slots vs TimeXer, TFT, TiDE, DUET, CrossLinear, PatchTST, etc., and stays competitive without future exo and under partial exo masking.[^src-gcgnet]

## Relation to Sibling Work

Same lab line as [[kite|KITE]] (probabilistic FM + exogenous) and [[dag|DAG]] (dual correlation injection). GCGNet is **deterministic** graph-consistent generation rather than flow-matching density estimation; it stresses **joint graph alignment** over two-step attention fusion used by [[source-timexer|TimeXer]] / [[source-exotst|ExoTST]].[^src-gcgnet]

## Links

- Source: [[source-gcgnet]]
- Concept: [[joint-temporal-channel-correlation]]
- Techniques: [[graph-structure-aligner]], [[graph-refiner]], [[variational-generator-exogenous]]
- Related: [[source-timexer]], [[source-crosslinear]], [[crosslinear]], [[source-exost]], [[source-kite]], [[source-dag]], [[patchtst]]

---

[^src-gcgnet]: [[source-gcgnet]]
