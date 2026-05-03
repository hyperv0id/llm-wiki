---
title: "OpenCity"
type: entity
tags:
  - spatio-temporal
  - foundation-model
  - traffic-prediction
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# OpenCity

OpenCity is a spatio-temporal foundation model for traffic prediction, serving as a key baseline for cross-city zero-shot transfer evaluation in the spatio-temporal forecasting literature[^src-most].

## Context

OpenCity represents the category of single-modal spatio-temporal foundation models that rely exclusively on numerical time series data. It is contrasted with multi-modal approaches like [[most|MoST]] that incorporate additional modalities (e.g., POI, weather, event text) for improved zero-shot transfer.

## Related

- [[most]] — MoST, multi-modal ST foundation model that outperforms OpenCity in zero-shot settings
- [[spatio-temporal-foundation-model]] — overview of ST foundation model paradigm

[^src-most]: [[source-most]]
