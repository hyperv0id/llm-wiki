---
title: "Spatio-Temporal Foundation Model"
type: concept
tags:
  - foundation-model
  - spatial-temporal
  - zero-shot
  - generalization
created: 2026-05-03
last_updated: 2026-05-12
source_count: 2
confidence: high
status: active
---

# Spatio-Temporal Foundation Model

A **spatio-temporal foundation model** is a large-scale pre-trained model designed for cross-city/cross-domain spatio-temporal prediction without requiring per-dataset training or fine-tuning[^src-most]. Unlike task-specific spatio-temporal models (e.g., STGCN, GWN) that are trained and evaluated on the same dataset, foundation models aim to capture universal spatio-temporal patterns transferable to unseen cities[^src-most].

## Motivation

Traditional spatio-temporal models face two deployment barriers[^src-most]:
1. **High cost**: Each new city requires collecting data, training a model from scratch, and tuning hyperparameters
2. **Poor generalization**: Models overfit to specific sensor topologies and fail on cities with different spatial configurations

Foundation models address both by pre-training once on diverse multi-city data and enabling zero-shot prediction on unseen cities[^src-most].

## Existing Models

### Single-Modal
- **[[opencity|OpenCity]]** (2024): Transformer + GNN architecture for traffic prediction. Trained on multiple cities, supports zero-shot inference but limited to single-modal time series data[^src-most].
- **[[urbandit|UrbanDiT]]** (NeurIPS 2025): Diffusion Transformer (DiT) backbone with prompt learning. Supports bi-directional prediction, temporal interpolation, spatial extrapolation, and spatio-temporal imputation with strong zero-shot capabilities[^src-urbandit].
- **UniST** (2024): Grid-based spatio-temporal foundation model for Euclidean-grid data[^src-most].
- **UrbanGPT** (2024): LLM-based spatio-temporal model using question-answering paradigm. Processes one sensor at a time, making it computationally expensive (7B parameters, 174s inference)[^src-most].
- **Pangu-Weather / Fengwu**: Weather-specific foundation models on Euclidean grids[^src-most].

### Multi-Modal
- **[[most|MoST]]** (KDD 2026): First multi-modality spatio-temporal foundation model. Supports satellite imagery, POI text, location, and time series as input modalities with adaptive SNR-based selection[^src-most].

## Key Challenges

1. **Modality variability**: Different cities have different available modalities with varying quality[^src-most]
2. **Spatial heterogeneity**: Local spatial patterns are highly region-specific and resist uniform modeling[^src-most]
3. **Scalability**: Must handle thousands of sensors efficiently[^src-most]

## Comparison with Time Series Foundation Models

| Dimension | ST Foundation Models | TS Foundation Models ([[timesfm|TimesFM]], [[chronos|Chronos]]) |
|-----------|---------------------|------------------------------------------------------------------|
| Input structure | Graph (sensors + topology) | Independent time series |
| Spatial modeling | Explicit (GNN, attention, experts) | None (channel-independent) |
| Modalities | Multi (image, text, location) | Single (numerical TS) |
| Primary task | Traffic/weather prediction | General forecasting |

## Related Pages

- [[most]] — MoST, first multi-modality ST foundation model
- [[urbandit]] — UrbanDiT, diffusion transformer for open-world spatiotemporal prediction
- [[traffic-forecasting]] — general traffic prediction
- [[multimodal-time-series-forecasting]] — multimodal TS forecasting
- [[large-scale-spatial-temporal-graph]] — large-scale ST graph challenges

[^src-most]: [[source-most]]
[^src-urbandit]: [[source-urbandit]]