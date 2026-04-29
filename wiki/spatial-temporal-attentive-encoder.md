---
title: "Spatial-Temporal Attentive Encoder"
type: technique
tags:
  - spatial-temporal
  - attention
  - encoder
  - traffic-forecasting
created: 2026-04-27
last_updated: 2026-04-27
source_count: 1
confidence: medium
status: active
---

# Spatial-Temporal Attentive Encoder (STAE)

The Spatial-Temporal Attentive Encoder (STAE) is the dual-pathway encoder in [[hyperd|HyperD]] that processes short-term and long-term periodicity through separate but architecturally similar pathways[^src-hyperd-hybrid-periodicity-decoupling]. It builds on the STAEformer-style spatial-temporal embedding with attention.

## Architecture

STAE comprises two parallel pathways[^src-hyperd-hybrid-periodicity-decoupling]:

### Pathway 1: Short-Term Encoder (STFE)
- **Input:** Original traffic signal X + short-term periodic embedding H_s (from [[frequency-aware-residual-representation|FR]])
- **Focus:** Local temporal dynamics — recent hours, abrupt changes
- **Output:** Short-term representation Z_s ∈ ℝ^{T'×N×D}

### Pathway 2: Long-Term Encoder (FR-STFE)
- **Input:** Residual signal R (from FR) + long-term periodic embedding H_l
- **Focus:** Global cyclical patterns — daily/weekly rhythms
- **Output:** Long-term representation Z_l ∈ ℝ^{T'×N×D}

Each pathway uses three embedding types[^src-hyperd-hybrid-periodicity-decoupling]:
- **Temporal Embedding** — captures time-of-day and day-of-week patterns
- **Spatial Embedding** — captures sensor identity
- **Spatial-Temporal Attention** — models interactions between all (N × T) tokens

## Design Rationale

The key insight is **input separation**: by feeding the original signal (with full frequency content) to the short-term encoder and the residual signal (with short-term fluctuations removed) to the long-term encoder, each pathway specializes in its target periodicity scale without mutual interference[^src-hyperd-hybrid-periodicity-decoupling].

## Comparison

Unlike single-stream architectures (standard STAEformer, DCRNN's encoder-decoder, or Transformer-based models), STAE's dual-pathway design explicitly allocates representational capacity across periodic scales[^src-hyperd-hybrid-periodicity-decoupling]. The [[dual-view-alignment-loss|DVA loss]] ensures the two pathways produce consistent-yet-complementary representations.

[^src-hyperd-hybrid-periodicity-decoupling]: [[source-hyperd-hybrid-periodicity-decoupling]]
