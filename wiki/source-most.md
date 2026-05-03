---
title: "MoST: A Foundation Model for Multi-modality Spatio-temporal Traffic Prediction"
type: source-summary
tags:
  - traffic-forecasting
  - foundation-model
  - multi-modality
  - spatio-temporal
  - kdd-2026
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# MoST: A Foundation Model for Multi-modality Spatio-temporal Traffic Prediction

**Authors**: Ronghui Xu, Jihao Chen, Jingdong Tian, Chenjuan Guo, Bin Yang (East China Normal University)
**Venue**: KDD 2026
**DOI**: 10.1145/3770854.3780162

## Core Contribution

MoST is the first foundation model capable of performing multi-modality spatio-temporal traffic prediction using any available modality inputs[^src-most]. Unlike prior spatio-temporal foundation models (e.g., OpenCity) that are limited to single-modal time series data, MoST leverages satellite imagery, POI text, and location coordinates as background context to enhance cross-city generalization.

## Key Innovations

### Multi-Modality Refinement Module
Encodes available modalities (satellite images via ResNet50, POI text via BERT, location coordinates, time series via patch-based RoPE encoder) and adaptively masks modalities with low signal-to-noise ratios (SNR)[^src-most]. Inspired by Shannon's second law, each modality is treated as an information channel; modalities with low SNR are deactivated via a Gumbel-Sigmoid gating mechanism. A contrastive learning objective guides the selector to suppress noisy modalities.

### Multi-Modality-Guided Spatial Expert Mechanism
Uses multi-modality context to dynamically select specialized spatial experts for different regions[^src-most]. Two types of experts are employed: modality-shared experts (one per activated modality) and routed experts (selected via a router from the modality-fused embedding). Each spatial expert models interactions between a sensor and its top-k nearest neighbors via cross-attention, capturing localized spatial dependencies efficiently.

### Non-Autoregressive Decoder
A Transformer decoder integrates temporal embeddings with spatially-aware patch embeddings for parallel multi-step prediction[^src-most].

## Experimental Results

Evaluated on five real-world datasets (Taxi-NYC, Taxi-CHI from UCTB; SD, GBA, GLA from LargeST) under zero-shot, few-shot, and full-shot settings[^src-most]. In zero-shot, MoST outperforms all baselines including the OpenCity foundation model and most full-shot end-to-end models. Ablation studies confirm that both the modality selector and spatial expert routing contribute significantly to performance.

## Limitations

- Requires pre-trained modality encoders (ResNet50, BERT) which are kept frozen
- Satellite imagery and POI data may not be available for all cities
- The Gumbel-Sigmoid threshold α is a fixed hyperparameter (0.8)
- Limited to four modalities; extension to more modalities is future work

[^src-most]: [[source-most]]