---
title: "VoT: Event-Driven Reasoning and Multi-Level Alignment for Time Series Forecasting"
type: source-summary
tags:
  - time-series
  - multimodal
  - llm
  - text-alignment
  - event-driven
  - iclr-2026
created: 2026-05-03
last_updated: 2026-05-03
source_count: 0
confidence: medium
status: active
---

# VoT: Event-Driven Reasoning and Multi-Level Alignment for Time Series Forecasting

**Source**: Siyuan Wang, Peng Chen, Yihang Wang, Wanghui Qiu, Chenjuan Guo, Bin Yang, Yang Shu (East China Normal University). arXiv:2603.15452. Accepted by ICLR 2026.

## Core Contribution

VoT (Value of Text) is a multimodal time series forecasting method that unlocks the predictive value of textual information through two complementary mechanisms: **Event-driven Reasoning** on exogenous text (news, policy documents) and **Multi-level Alignment** across both representation and prediction levels.

## Key Innovations

### 1. Event-driven Reasoning with Historical In-Context Learning (HIC)

A three-step generative pipeline (template generation → summarization → reasoning) uses LLMs to extract forecasting-relevant information from exogenous text. HIC enhances this by constructing a knowledge base of corrected reasoning samples during training and retrieving similar historical examples as error-informed guidance during inference — without requiring fine-tuning.

### 2. Multi-level Alignment

- **Representation level**: Endogenous Text Alignment (ETA) converts time series statistics into textual descriptions, then uses decomposed pattern extraction (trend/seasonal) and decomposed contrastive learningODIS to align text and time series representations.
- **Prediction level**: Adaptive Frequency Fusion (AFF) decomposes both event-driven and numerical predictions into frequency bands (low/mid/high) and learns adaptive fusion weights per band, achieving complementary advantages across modalities.

### 3. Dual-Branch Architecture

The event-driven prediction branch processes exogenous text via LLM reasoning, while the numerical prediction branch aligns endogenous text with time series. AFF fuses outputs from both branches.

## Experimental Results

Evaluated on 10 real-world multimodal datasets across diverse domains (Agriculture, Climate, Economy, Energy, Environment, Health, Security, Social Good, Traffic, Weather). VoT achieves 20/20 first-place counts against time series-only and text-enhanced baselines (PatchTST, iTransformer, RaFT and their text-enhanced variants), and outperforms multimodal methods including GPT4TS, GPT4MTS, TaTS, Time-VLM, and CALF.

## Limitations

- Relies on LLM inference for the event-driven branch, which may introduce latency
- HIC requires constructing and maintaining a knowledge base of corrected reasoning samples
- The approach assumes availability of both exogenous and endogenous text, which may not hold in all domains

## Related Pages

- [[vot]] — the VoT model entity
- [[event-driven-reasoning]] — event-driven reasoning concept
- [[historical-in-context-learning]] — HIC technique
- [[multi-level-alignment]] — multi-level alignment concept
- [[endogenous-text-alignment]] — ETA technique
- [[adaptive-frequency-fusion]] — AFF technique
- [[multimodal-time-series-forecasting]] — task concept
- [[mindts]] — related multimodal TS model (anomaly detection)
- [[fine-grained-time-text-semantic-alignment]] — MindTS's alignment technique (compare with ETA)
