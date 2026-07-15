---
title: "CrossLinear"
type: entity
tags:
  - time-series-forecasting
  - exogenous
  - linear-models
  - plug-and-play
  - kdd-2025
created: 2026-07-13
last_updated: 2026-07-16
source_count: 1
confidence: high
status: active
---

# CrossLinear

**CrossLinear** is a Linear-based model for **time series forecasting with exogenous variables** (many-to-one), proposed by Zhou et al. (USTC / Deqing Alpha Innovation Institute) at KDD 2025 (arXiv:2505.23116). Code: https://github.com/mumiao2000/CrossLinear.[^src-crosslinear]

## Problem

Predict future endogenous \(X^{\mathrm{endo}}_{T+1:T+S}\) from historical endogenous and exogenous series without forecasting exogenous targets. Full CD dependency models overfit time-varying/spurious relations; pure CI models cannot inject exogenous signal into the target path.[^src-crosslinear]

## Architecture

| Stage | Role |
|-------|------|
| RevIN | Instance norm / de-norm for non-stationarity |
| [[cross-correlation-embedding\|Cross-correlation embedding]] | 1D conv over stacked endo+exo; residual mix with learnable \(\alpha\) |
| Patch + PE | Short-term temporal tokens + \(\beta\)-weighted positional embedding |
| Linear head | Global projection for long-term patterns |

Overall complexity \(O(T)\). Multivariate forecasting reuses the module with multi-channel cross-correlation outputs and shared patch/head weights.[^src-crosslinear]

## Empirical Snapshot

On 12 exogenous benchmarks (ECL, Weather, ETT, Traffic, EPF markets), CrossLinear outperforms [[source-timexer|TimeXer]] on most settings while being faster to train; the embedding is plug-and-play for SparseTSF, RLinear, PatchTST, DLinear, Autoformer, with largest gains on high-exo datasets (ECL 320, Traffic 861).[^src-crosslinear]

## Place in the Exogenous Lineage

- vs [[source-timexer|TimeXer]]: attention dual-granularity (patch endo + variate exo) → heavier \(O((T/p)^2)\); CrossLinear is linear plug-in residual fusion.
- vs [[source-exotst|ExoTST]]: past/future exo as separate modalities + cross-temporal fusion; CrossLinear uses historical endo/exo only under TimeXer-style setup.
- vs [[source-exost|ExoST]]: ST select-then-balance for past/future exo types; CrossLinear is non-ST series many-to-one with convolution residual mix.
- vs [[source-exollm|ExoLLM]]: LLM multi-grained text prompts; CrossLinear is purely numerical Linear.
- vs [[source-gcgnet|GCGNet]]: GCGNet cites CrossLinear as a channel-then-temporal two-step baseline and proposes joint graph-consistent generation instead.[^src-crosslinear]

## Links

- Source: [[source-crosslinear]]
- Technique: [[cross-correlation-embedding]]
- Related: [[source-timexer]], [[source-exost]], [[source-exollm]], [[source-exotst]], [[source-gcgnet]], [[channel-independence]], [[patch-based-tokenization]]

---

[^src-crosslinear]: [[source-crosslinear]]
