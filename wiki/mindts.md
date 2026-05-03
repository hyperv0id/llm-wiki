---
title: "MindTS"
type: entity
tags:
  - multimodal-time-series
  - anomaly-detection
  - time-series
  - text-encoding
  - iclr-2026
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# MindTS

**MindTS** (Multimodal Time Series Anomaly Detection with Semantic Alignment and Condensed Interaction) is a model proposed by Hu, Jin, Shu, Chen, Yang, and Guo from East China Normal University, published at ICLR 2026[^src-multimodal-ts-anomaly-detection]. It is the first dedicated multimodal anomaly detection model that jointly leverages time series data and text modalities.

## Core Problem

Traditional time series anomaly detection methods (e.g., DCdetector, GDN, Anomaly Transformer) are unimodal, relying solely on numerical data. However, real-world scenarios often provide rich textual context (news reports, weather forecasts, policies) that could improve anomaly detection. Two key challenges exist:

1. **Semantic alignment**: Text and time series reside in different semantic spaces, making alignment difficult
2. **Redundant information**: Text contains lengthy descriptions irrelevant to anomaly detection, which can degrade performance if naively fused[^src-multimodal-ts-anomaly-detection]

## Architecture

MindTS consists of two main modules:

### 1. Fine-grained Time-text Semantic Alignment

- Splits text into **endogenous** (statistical summaries per patch) and **exogenous** (shared background context) views
- Cross-view text fusion: uses endogenous as query, exogenous as key/value
- Multimodal alignment via contrastive learning: pulls aligned time-text pairs closer in embedding space[^src-multimodal-ts-anomaly-detection]

### 2. Content Condenser Reconstruction

- Based on Information Bottleneck principle: compresses aligned text by minimizing mutual information
- Bernoulli sampling with straight-through estimator generates binary masks
- Condensed text reconstructs **masked** time series (≈50% random masking)
- Smoothness loss ensures stable compression across adjacent patches[^src-multimodal-ts-anomaly-detection]

## Performance

MindTS outperforms 17 baselines across 6 real-world datasets (Weather, Energy, Environment, KR, EWJ, MDT):

- **Weather**: Aff-F 82.66 vs. best baseline 81.06
- **Energy**: Aff-F 74.37 vs. 70.81
- **MDT**: Aff-F 89.19 vs. 80.81 (largest improvement)[^src-multimodal-ts-anomaly-detection]

The model is robust to different LLMs (DeepSeek, GPT2, BERT, LLAMA), and the content condenser's compression strength μ is stable across 0.1–0.9.

## Comparison with Related Models

| Model | Task | Modality | Key Technique |
|-------|------|----------|---------------|
| **MindTS** | Anomaly detection | Time series + text | Cross-view alignment + content condenser |
| [[most|MoST]] | Traffic prediction | Multi-modal (image+text+location+TS) | SNR-based modality selection |
| [[channelmts|ChannelMTS]] | HSR channel prediction | Time series + environmental features | Retrieval-augmented statistical channel |
| [[unica|UniCA]] | TSFM covariate adaptation | Time series + heterogeneous covariates | Covariate homogenization + dual fusion |
| DCdetector | Anomaly detection | Time series only | Contrastive learning |

MindTS is the only model specifically designed for **anomaly detection** with **text** input, whereas most other multimodal TS models focus on forecasting/prediction.

## Related Pages

- [[source-multimodal-ts-anomaly-detection]] — source summary
- [[multimodal-time-series-anomaly-detection]] — task concept
- [[fine-grained-time-text-semantic-alignment]] — alignment technique
- [[content-condenser-reconstruction]] — redundancy filtering technique
- [[multimodal-time-series-forecasting]] — related multimodal TS task
- [[vot]] — VoT model (same lab: ECNU, forecasting task)
- [[endogenous-text-alignment]] — VoT's ETA technique (compare with MindTS's alignment)
- [[aurora]] — Aurora generative multimodal TS foundation model (different task: forecasting vs. anomaly detection)
- [[mutual-information]] — information theory concept used in condenser

[^src-multimodal-ts-anomaly-detection]: [[source-multimodal-ts-anomaly-detection]]
