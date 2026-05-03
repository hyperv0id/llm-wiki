---
title: "VoT (Value of Text)"
type: entity
tags:
  - multimodal-time-series
  - forecasting
  - llm
  - text-alignment
  - event-driven
  - iclr-2026
created: 2026-05-03
last_updated: 2026-05-03
source_count: 2
confidence: high
status: active
---

# VoT (Value of Text)

**VoT** (Value of Text) is a multimodal time series forecasting model proposed by Wang et al. from East China Normal University, accepted at ICLR 2026[^src-event-driven-ts-forecasting]. It is the first method to jointly leverage LLMs for both feature extraction and reasoning while supporting both exogenous and endogenous text types for comprehensive multimodal forecasting.

## Core Problem

Existing multimodal time series forecasting methods either use LLMs only for feature extraction (e.g., GPT4TS, TimeLLM, CALF) or handle only one text type — endogenous (statistical summaries) or exogenous (news, policy documents)[^src-event-driven-ts-forecasting]. VoT addresses two key challenges:

1. **Insufficient text utilization**: Methods using endogenous text extract information already present in the time series; methods using exogenous text focus only on representation-level fusion, leaving deep semantic information untapped[^src-event-driven-ts-forecasting].
2. **Modality gap**: Text provides guidance for event-driven dynamics while time series captures subtle numerical trends, but the considerable gap between modalities prevents effective cross-modal integration[^src-event-driven-ts-forecasting].

## Architecture

VoT employs a dual-branch architecture:

### Event-Driven Prediction Branch

Processes exogenous text through a three-step generative pipeline:
1. **Template generation**: LLM creates a dataset-specific structured dictionary
2. **Summarization**: Filters redundant information while preserving forecasting-relevant content
3. **Reasoning**: A reasoning-focused LLM generates numerical predictions and explanatory reasoning processes

Enhanced by **[[historical-in-context-learning|Historical In-Context Learning (HIC)]]**, which retrieves corrected historical reasoning examples as error-informed guidance during inference[^src-event-driven-ts-forecasting].

### Numerical Prediction Branch

Aligns endogenous text with time series via **[[endogenous-text-alignment|Endogenous Text Alignment (ETA)]]**, which uses decomposed pattern extraction (trend/seasonal) and decomposed contrastive learning for representation-level alignment[^src-event-driven-ts-forecasting].

### Fusion

**[[adaptive-frequency-fusion|Adaptive Frequency Fusion (AFF)]]** decomposes both branch predictions into frequency bands and learns adaptive fusion weights, achieving complementary advantages at the prediction level[^src-event-driven-ts-forecasting].

## Performance

VoT achieves state-of-the-art results on 10 real-world multimodal datasets, with 20/20 first-place counts against time series-only and text-enhanced baselines. It outperforms multimodal methods including GPT4TS, GPT4MTS, TaTS, Time-VLM, and CALF[^src-event-driven-ts-forecasting].

## Comparison with Related Models

| Model | Task | LLM Usage | Text Types | Key Innovation |
|-------|------|-----------|------------|----------------|
| **VoT** | Forecasting | Feature extraction + Reasoning | Exogenous + Endogenous | Event-driven reasoning + Multi-level alignment |
| [[tats|TaTS]] | Forecasting + Imputation | Feature extraction | Time-paired texts | Text as auxiliary variables (plug-and-play) |
| [[mindts|MindTS]] | Anomaly detection | Feature extraction | Exogenous + Endogenous | Cross-view alignment + Content condenser |
| [[most|MoST]] | Traffic prediction | Feature extraction | Multi-modal (image+text+location+TS) | SNR-based modality selection |
| [[unica|UniCA]] | TSFM covariate adaptation | Feature extraction | Heterogeneous covariates | Covariate homogenization |
| [[aurora|Aurora]] | Forecasting | Feature extraction | Multi-modal (text+image+TS) | Modality-Guided Attention + Flow Matching |
| GPT4TS | Forecasting | Feature extraction | Endogenous only | LLM as time series encoder |

VoT is unique in using LLMs for **reasoning** (not just feature extraction) and supporting **both** exogenous and endogenous text simultaneously[^src-event-driven-ts-forecasting]. In contrast, [[tats|TaTS]] achieves multimodal integration through a simpler plug-and-play approach — treating text as auxiliary variables without LLM reasoning or architecture modification[^src-language-in-the-flow-of-time].

## Related Pages

- [[source-event-driven-ts-forecasting]] — source summary
- [[event-driven-reasoning]] — event-driven reasoning concept
- [[historical-in-context-learning]] — HIC technique
- [[multi-level-alignment]] — multi-level alignment concept
- [[endogenous-text-alignment]] — ETA technique
- [[adaptive-frequency-fusion]] — AFF technique
- [[multimodal-time-series-forecasting]] — task concept
- [[mindts]] — related multimodal TS model (same lab: ECNU)
- [[aurora]] — Aurora generative multimodal TS foundation model (different paradigm: generative vs. LLM reasoning)
- [[tats]] — TaTS plug-and-play multimodal framework (simpler alternative approach)

[^src-event-driven-ts-forecasting]: [[source-event-driven-ts-forecasting]]
[^src-language-in-the-flow-of-time]: [[source-language-in-the-flow-of-time]]
