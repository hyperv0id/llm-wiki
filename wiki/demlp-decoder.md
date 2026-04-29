---
title: "Decoupled MLP Decoder (fDMLP)"
type: technique
tags:
  - decoder
  - mlp
  - trend-removal
  - traffic-forecasting
created: 2026-04-27
last_updated: 2026-04-27
source_count: 1
confidence: medium
status: active
---

# Decoupled MLP Decoder (fDMLP)

The Decoupled MLP Decoder (fDMLP) is the two-stage coarse-to-fine decoder in [[hyperd|HyperD]], inspired by DLinear's decomposition philosophy but adapted for multi-scale periodicity fusion[^src-hyperd-hybrid-periodicity-decoupling].

## Two-Stage Architecture

### Stage 1: De-trending MLP (DMLP)

- Receives the concatenated representations [Z_s || Z_l] from the [[spatial-temporal-attentive-encoder|STAE]] dual-pathway encoder
- Projects to a **coarse prediction** Ŷ_coarse
- Simultaneously extracts a **trend component** T that captures monotonic directional bias
- The trend is **removed** from the coarse prediction to prevent over-smoothing toward the mean[^src-hyperd-hybrid-periodicity-decoupling]

### Stage 2: Fine-grained MLP (fDMLP)

- Takes the de-trended coarse prediction as input
- Applies channel-wise MLP layers with **residual connections**
- Recovers fine-grained spatial-temporal details
- Produces the **final prediction** Ŷ[^src-hyperd-hybrid-periodicity-decoupling]

## Decoupling Strategy

The core idea parallels DLinear's decomposition (separating trend from seasonal components), but here the trend extraction serves a different purpose: preventing the decoder from collapsing to a simple averaging solution when multiple periodic signals are fused[^src-hyperd-hybrid-periodicity-decoupling]. The two-stage coarse-to-fine process mirrors [[hyperd|HyperD]]'s overall multi-scale philosophy.

## Impact

Replacing fDMLP with a simple MLP decoder increases MAE by +2.8%, demonstrating that the two-stage decoupled design is essential for effectively fusing the dual-pathway encoder outputs[^src-hyperd-hybrid-periodicity-decoupling].

[^src-hyperd-hybrid-periodicity-decoupling]: [[source-hyperd-hybrid-periodicity-decoupling]]
