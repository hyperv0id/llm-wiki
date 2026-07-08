---
title: "Source: Solar-VLM — Multimodal Vision-Language Models for Augmented Solar Power Forecasting"
type: source-summary
tags:
  - spatiotemporal
  - multimodal-vlm
  - solar-energy
  - 2026
created: 2026-07-07
last_updated: 2026-07-07
source_count: 1
confidence: medium
status: active
---

# Source: Solar-VLM — Multimodal Vision-Language Models for Augmented Solar Power Forecasting

**Authors**: Hang Fan, Haoran Pei, Runze Liang, Weican Liu, Long Cheng, Wei Wei (North China Electric Power University / Tsinghua University / Nanyang Technological University). **Venue**: arXiv:2604.04145 (Apr 2026)[^src-solar-vlm].

## Core Idea

Solar-VLM proposes a **unified multimodal vision-language-time-series framework** for multi-site photovoltaic (PV) power forecasting. Unlike prior work that treats time-series, satellite imagery, and weather text in isolation, Solar-VLM jointly fuses all three modalities through modality-specific encoders, a graph-based spatial dependency module, and a cross-site attention mechanism[^src-solar-vlm]. The framework leverages a **frozen Qwen-based VLM** as the backbone for both visual and text encoding, benefiting from pretrained foundation model knowledge[^src-solar-vlm].

## Method

1. **Modality-Specific Encoders**: The time-series encoder uses a **patch-based design** (patch length 10, stride 8) with dual-path local (memory bank retrieval) and global (multi-head self-attention) modeling, gated fusion, and inter-variable attention. The text encoder converts historical observations into structured key-value prompts (covering task config, spatiotemporal context, power status, meteorological conditions) encoded by a frozen Qwen-VLM text encoder with an auxiliary temporal encoding branch. The visual encoder processes the most recent 8 satellite images (128×128 pixels) via a frozen Qwen image encoder with temporal Transformer layers[^src-solar-vlm].

2. **Cross-Site Joint Modeling** (two-stage): **Stage 1 — Graph Learner** applies a **Graph Attention Network (GAT)** over a KNN graph (K=5) constructed from Haversine distances, with distance-based edge weights. This operates exclusively on time-series features since they remain physically comparable across sites. **Stage 2 — Cross-Site Attention** treats graph-enhanced temporal features as queries and fused multimodal features as keys/values, enabling adaptive data-driven information sharing across sites without imposing explicit structural constraints on heterogeneous multimodal embeddings[^src-solar-vlm].

3. **Multimodal Fusion & Prediction**: Per-site multimodal fusion concatenates temporal, textual, and visual representations through a two-layer MLP. The final prediction combines the cross-site attention output with a temporal-only auxiliary prediction via adaptive gating[^src-solar-vlm].

## Contributions

- First unified paradigm jointly integrating satellite imagery (visual), historical text (language), and generation data (time-series) for multi-site PV forecasting[^src-solar-vlm]
- **Cross-modal interaction** via a Qwen-VLM backbone + cross-modal attention + graph learner[^src-solar-vlm]
- **Retrieval-augmented long-term memory** via a dual-memory architecture (local + global) with a memory fusion gate[^src-solar-vlm]

## Results

Evaluated on 8 real-world PV stations in Hebei Province, China (14 features, 15-min intervals, 3-day input window). Solar-VLM outperforms 7 baselines (LSTM, Informer, FEDformer, TimesNet, TimeLLM, SUNSET, TimeVLM) across all horizons (T=3/6/12/24/36/48/96). Key gains: at T=24, MSE reduced by 3.5% and MAE by 10.5% vs second-best. At T=48, MSE reduced by 8.5% and MAE by 18.3%. Ablation confirms cross-site attention is the most critical component — its removal causes the largest degradation[^src-solar-vlm].

## Related Pages

- [[source-st-vision-llm]] — ST-Vision-LLM, a VLM-based framework for spatiotemporal traffic forecasting using visual encoding of traffic matrices
- [[source-gpt4mts]] — GPT for Multimodal Time Series (related multimodal LLM forecasting approach)

## Limitations

The optimal historical time-series length (L=192) and the optimal number of satellite images (k=8) require tuning per dataset. The KNN graph uses geographical proximity as a proxy for meteorological correlation, which may not fully capture complex weather system interactions. The Qwen-VLM backbone is frozen, limiting adaptation to domain-specific visual/textual features[^src-solar-vlm].

[^src-solar-vlm]: [[source-solar-vlm]]