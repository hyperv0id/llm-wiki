---
title: "Multi-Level Alignment for Multimodal Time Series"
type: concept
tags:
  - multimodal
  - time-series
  - alignment
  - contrastive-learning
  - frequency-fusion
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# Multi-Level Alignment for Multimodal Time Series

**Multi-level Alignment** is a concept introduced in [[vot|VoT]] (ICLR 2026) for fully integrating the predictive capabilities of time series and text modalities across both representation and prediction levels[^src-event-driven-ts-forecasting].

## Motivation

While text provides crucial guidance for sudden shifts and event-related dynamics, time series modeling captures subtle fluctuations and numerical trends that text cannot describe. However, the considerable modality gap between these modalities prevents existing methods from achieving effective cross-modal integration[^src-event-driven-ts-forecasting].

Multi-level alignment addresses this by performing alignment at two complementary levels:

## Two Levels

### 1. Representation-Level Alignment: Endogenous Text Alignment (ETA)

**[[endogenous-text-alignment|ETA]]** operates in the numerical prediction branch, establishing deep semantic alignment between temporal patterns and their textual representations[^src-event-driven-ts-forecasting]:

- Converts time series statistics (mean, frequency) into structured textual descriptions (endogenous text)
- Uses **decomposed pattern extraction**: dual-query attention extracts trend and seasonal information from text, then cross-attention aligns temporal representations with these textual components
- Uses **decomposed contrastive learning**: contrastive loss computed separately for trend and seasonal component pairs, pulling aligned time-text pairs closer in embedding space[^src-event-driven-ts-forecasting]

### 2. Prediction-Level Alignment: Adaptive Frequency Fusion (AFF)

**[[adaptive-frequency-fusion|AFF]]** fuses outputs from the event-driven prediction branch and numerical prediction branch[^src-event-driven-ts-forecasting]:

- Decomposes both predictions into frequency bands (low/mid/high) via FFT
- Learns adaptive fusion weights per frequency band, rather than using fixed ratios
- Enables domain-driven optimization: different datasets have varying dependencies on textual vs. numerical information across frequency bands[^src-event-driven-ts-forecasting]

## Comparison with MindTS's Alignment

| Dimension | VoT (Multi-level Alignment) | [[mindts|MindTS]] ([[fine-grained-time-text-semantic-alignment|Fine-grained Alignment]]) |
|-----------|---------------------------|------|
| Task | Forecasting | Anomaly detection |
| Text types | Exogenous + Endogenous | Exogenous + Endogenous |
| Alignment level | Representation + Prediction | Representation only |
| Alignment method | Decomposed contrastive (trend/seasonal) | Cross-view attention + contrastive |
| Text decomposition | Trend vs. seasonal components | Endogenous (per-patch) vs. exogenous (shared) |
| Prediction fusion | Adaptive frequency fusion (AFF) | Content condenser + reconstruction |

Both methods come from the same lab (ECNU) and share the idea of decomposing text into complementary views for alignment, but VoT extends alignment to the prediction level via frequency-domain fusion[^src-event-driven-ts-forecasting].

## Related Pages

- [[vot]] — the VoT model
- [[endogenous-text-alignment]] — ETA technique
- [[adaptive-frequency-fusion]] — AFF technique
- [[event-driven-reasoning]] — complementary reasoning paradigm
- [[fine-grained-time-text-semantic-alignment]] — MindTS's alignment (compare)
- [[multimodal-time-series-forecasting]] — task concept

[^src-event-driven-ts-forecasting]: [[source-event-driven-ts-forecasting]]
