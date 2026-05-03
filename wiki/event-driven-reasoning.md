---
title: "Event-Driven Reasoning for Time Series Forecasting"
type: concept
tags:
  - time-series
  - multimodal
  - llm
  - reasoning
  - event-driven
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# Event-Driven Reasoning for Time Series Forecasting

**Event-driven Reasoning** is a paradigm introduced in [[vot|VoT]] (ICLR 2026) that leverages the reasoning capabilities of LLMs to extract forecasting-relevant information from exogenous text (news, policy documents, social media) and generate numerical predictions[^src-event-driven-ts-forecasting].

## Motivation

Real-world time series exhibit abrupt changes driven by external events — such as unemployment spikes during the 2008 financial crisis or the COVID-19 pandemic — that are difficult to predict from historical numerical patterns alone[^src-event-driven-ts-forecasting]. Textual information (news, policy announcements) describes these events, but existing multimodal methods either:

- Use endogenous text (statistical summaries) that largely overlaps with information already in the time series
- Use exogenous text only for representation-level fusion, failing to extract deep semantic reasoning about how events affect future values[^src-event-driven-ts-forecasting]

## Three-Step Generative Pipeline

VoT's event-driven reasoning processes exogenous text through:

1. **Template Generation**: LLM creates a dataset-specific structured dictionary (key-value mappings) based on dataset description and exogenous text-time series samples, guiding extraction of predictive information[^src-event-driven-ts-forecasting].

2. **Summarization**: Using the template, LLM generates summaries from raw exogenous text and time series, filtering redundant information while preserving content influential to forecasting[^src-event-driven-ts-forecasting].

3. **Reasoning**: A specialized reasoning LLM (Reasoner) processes summaries and time series to generate both numerical predictions and explanatory reasoning processes[^src-event-driven-ts-forecasting].

## Historical In-Context Learning (HIC)

The basic pipeline operates in an unsupervised manner, which may introduce suboptimal guidance. **[[historical-in-context-learning|HIC]]** addresses this by:

- **Training phase**: Correcting reasoning with ground truth and storing corrected samples in a knowledge base
- **Inference phase**: Retrieving the most similar historical example and integrating it into the reasoning prompt as error-informed guidance[^src-event-driven-ts-forecasting]

This enables LLMs to learn from past errors without expensive fine-tuning.

## Comparison with Other LLM-Based TS Methods

| Method | LLM Role | Text Type | Reasoning |
|--------|----------|-----------|-----------|
| **VoT** | Feature extraction + Reasoning | Exogenous + Endogenous | Yes |
| GPT4TS | Feature extraction | Endogenous | No |
| TimeLLM | Feature extraction | Endogenous | No |
| CALF | Feature extraction | Endogenous | No |
| GPT4MTS | Feature extraction | Exogenous | No |
| TaTS | Feature extraction | Exogenous | No |

VoT is the first method to use LLMs for **reasoning** (not just feature extraction) in multimodal time series forecasting[^src-event-driven-ts-forecasting].

## Related Pages

- [[vot]] — the VoT model
- [[historical-in-context-learning]] — HIC technique
- [[multi-level-alignment]] — complementary alignment approach
- [[multimodal-time-series-forecasting]] — task concept
- [[mindts]] — related multimodal TS model (anomaly detection, same lab)

[^src-event-driven-ts-forecasting]: [[source-event-driven-ts-forecasting]]
