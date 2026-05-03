---
title: "MoST"
type: entity
tags:
  - traffic-forecasting
  - foundation-model
  - multi-modality
  - spatio-temporal
  - kdd-2026
created: 2026-05-03
last_updated: 2026-05-03
source_count: 2
confidence: high
status: active
---

# MoST

**MoST** (Multi-Modality Spatio-temporal Traffic Prediction) is a foundation model for multi-modality spatio-temporal traffic prediction, proposed by Xu, Chen, Tian, Guo, and Yang from East China Normal University at KDD 2026[^src-most]. It is the first foundation model capable of performing cross-city traffic prediction using any available modality inputs, including satellite imagery, points of interest (POI), location coordinates, and time series data.

## Architecture

MoST consists of two core modules:

### 1. Multi-Modality Refinement Module

Encodes and filters available modalities to retain only high-quality information[^src-most]:

- **Modality-specific encoders**: ResNet50 for satellite imagery, BERT for POI text, normalized coordinates for location, and a patch-based RoPE encoder for time series
- **SNR-based modality selector**: Estimates signal-to-noise ratio (SNR) for each modality using uncertainty and relevance predictors, then applies Gumbel-Sigmoid gating to deactivate low-SNR channels
- **Contrastive learning**: Guides the selector by ensuring modality-noised representations are close to modality-dropped representations while remaining distinct from clean representations

### 2. Spatio-Temporal Prediction Module

Leverages refined multi-modality context to capture region-specific spatial patterns[^src-most]:

- **Multi-modality-guided spatial gather block**: Activates modality-shared experts (one per activated modality) and routed experts (selected via a router) to model local spatial dependencies
- **Spatial experts**: Each expert uses cross-attention between a sensor and its top-k nearest neighbors, reweighted by adjacency weights
- **Non-autoregressive Transformer decoder**: Integrates temporal and spatial embeddings for parallel multi-step prediction

## Key Results

In zero-shot evaluation across five datasets (Taxi-NYC, Taxi-CHI, SD, GBA, GLA), MoST outperforms:
- All heuristic baselines (HA, VAR)
- Most full-shot end-to-end models (STGCN, GWN, ASTGCN, STWA, BigST, PatchSTG)
- The single-modal foundation model OpenCity

Ablation studies confirm that removing the modality selector, spatial expert routing, or multi-modality data each cause significant performance degradation[^src-most].

## Comparison with Related Models

| Model | Type | Modality | Generalization |
|-------|------|----------|----------------|
| **MoST** | Foundation model | Multi-modal (image + text + location + TS) | Zero-shot cross-city |
| [[conformer|ConFormer]] | Task-specific | Single-modal + accident data | Full-shot only |
| [[igstgnn|IGSTGNN]] | Task-specific | Single-modal + incident data | Full-shot only |
| OpenCity | Foundation model | Single-modal (TS only) | Zero-shot cross-city |
| [[unica|UniCA]] | Adaptation framework | Multi-modal (adapts TSFMs) | Zero-shot via TSFM |
| [[aurora|Aurora]] | Foundation model | Multi-modal (text + image + TS) | Zero-shot cross-dataset |

MoST is unique in being a **native multi-modality foundation model** for spatio-temporal prediction, rather than adapting existing single-modal TSFMs (like UniCA) or being task-specific (like ConFormer/IGSTGNN)[^src-most]. [[aurora|Aurora]] is a related multi-modal foundation model but focuses on general time series forecasting (not spatio-temporal) and uses a generative Flow Matching approach rather than MoST's discriminative approach[^src-aurora].

## Related Pages

- [[source-most]] — source summary
- [[multi-modality-refinement]] — SNR-based modality selection technique
- [[multi-modality-guided-spatial-expert]] — spatial expert mechanism
- [[spatio-temporal-foundation-model]] — ST foundation model concept
- [[traffic-forecasting]] — general traffic prediction
- [[multimodal-time-series-forecasting]] — multimodal TS forecasting
- [[mixture-of-experts]] — MoE architecture used in spatial expert routing
- [[mindts]] — MindTS multimodal anomaly detection (prediction vs. anomaly detection contrast)
- [[multimodal-time-series-anomaly-detection]] — multimodal TS anomaly detection task
- [[aurora]] — Aurora generative multimodal TS foundation model

[^src-most]: [[source-most]]
[^src-aurora]: [[source-aurora]]