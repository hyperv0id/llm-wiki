---
title: "CRAFT — Cross City Traffic Flow Generation via Retrieval Augmented Diffusion Model"
type: source-summary
tags:
  - traffic-flow
  - diffusion-model
  - retrieval-augmented-generation
  - cross-city-transfer
  - generative-model
created: 2026-06-08
last_updated: 2026-08-30
source_count: 2
confidence: high
status: active
---

# CRAFT — Cross City Traffic Flow Generation via Retrieval Augmented Diffusion Model

**Authors**: Yudong Li, Jingyuan Wang*, Xie Yu, Peiyu Wang (Beihang University), Qian Huang (Huawei)  
**Venue**: NeurIPS 2025 (39th Conference on Neural Information Processing Systems)  
**Code**: [github.com/lyd1881310/CRAFT](https://github.com/lyd1881310/CRAFT)

## Summary

CRAFT is a DDPM-based diffusion generation model designed for **zero-shot cross-city traffic flow generation** — synthesizing realistic traffic flow for a new city without any historical flow data from that city. It is trained on multiple source cities and directly deployed to unseen target cities using only publicly available geographic features (POIs, roads, population).

## Core Problem

Traditional traffic flow generation models require city-specific historical data for training, limiting deployment in cities with scarce records. Collecting traffic flow data faces high costs and privacy constraints. CRAFT enables cross-city transfer by learning from common geographic contexts shared across cities.

## Two Key Challenges & Solutions

1. **Domain Shift**: In cross-city settings, regions with similar geographic features may exhibit markedly different traffic flow patterns. CRAFT addresses this via **Geographic Feature Alignment (GFA)** — two complementary losses: Traffic Flow Alignment (TFA) aligns geographic representations with actual flow patterns within source cities, and Cross-City Alignment (CCA) uses optimal transport to project semantically similar regions across cities into proximity.

2. **Insufficient Condition**: Static geographic features alone cannot capture temporal dynamics (stochastic properties like mean, peak, variance). CRAFT addresses this via **Retrieval-based Condition Augmentation (RCA)** — retrieving similar historical flow segments from source cities based on geographic representation similarity and time information (month/day/hour), then fusing them as augmented conditions via self-attention.

## Architecture

- **Backbone**: DDPM with 1D-U-Net noise estimator
- **GFA module**: Basic geo-features (POIs via TF-IDF + road length + population) → MLP → Graph Transformer spatial encoder → TFA + CCA losses
- **RCA module**: Time embedding (month/day/hour) + retrieved top-K flow sequences from source cities → self-attention aggregation → MLP → condition vector
- Both GFA and RCA are lightweight plug-in components requiring no backbone modifications

## Key Results

- **4 bicycle trip datasets**: Chicago (CHI), Washington D.C. (DC), Toronto (TRT), New York City (NYC); leave-one-city-out cross-validation
- **SOTA zero-shot performance**: 59.7% improvement over the average of all baselines, 22.5% over second-best (GMEL), 61.5% over ordinary DDPM
- **Downstream utility**: Synthetic data from CRAFT trains LSTM/Transformer predictors with only 10.4% avg degradation vs. real data (min 3.8%, max 22.2%)
- **Ablation**: GFA provides the most improvement (domain shift is the major problem); within RCA, temporal embedding contributes most, followed by retrieved flow features
- **TFA + CCA**: TFA aligns geo-features with flow patterns; CCA mitigates cross-city distribution shift (t-SNE analysis confirms)
- **Sensitivity**: Model robust across hyperparameters (max 8.8% fluctuation); stable performance across temporal horizons up to T=168
- **Baselines**: GMEL, DFG, KSTDiff, CGAN, Diffwave, DiT, DDPM, CVAE

## Limitations

- Validated only on in-out flow generation; origin-destination (OD) flows not explored
- Temporal horizon limited to 168 steps due to computational constraints
- Experimental setup: bicycle-sharing systems in four North American cities; broader geographic diversity untested

## Relevance to Wiki

CRAFT is the first work explicitly targeting cross-city zero-shot traffic flow generation with a retrieval-augmented diffusion framework. It introduces the concepts of geographic feature alignment and retrieval-based condition augmentation for cross-city transfer, and connects to the broader RAG-for-spatio-temporal paradigm alongside [[rast|RAST]] (AAAI 2026). The [[jingyuan-wang|Jingyuan Wang]] lab at Beihang University is also responsible for BIGCity, GTG, [[pdformer|PDFormer]], and HiFiNet; PDFormer 原文可核实其通讯作者即 Jingyuan Wang（北航），共同一作为 Jiawei Jiang 与 Chengkai Han[^src-pdformer-jiang-2023]。

[^src-pdformer-jiang-2023]: [[source-pdformer-jiang-2023]]
