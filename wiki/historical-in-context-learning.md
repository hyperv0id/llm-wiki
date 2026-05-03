---
title: "Historical In-Context Learning (HIC)"
type: technique
tags:
  - llm
  - in-context-learning
  - retrieval
  - error-correction
  - time-series
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# Historical In-Context Learning (HIC)

**Historical In-Context Learning (HIC)** is a technique introduced in [[vot|VoT]] (ICLR 2026) that synergizes historical reasoning information with In-Context Learning (ICL) to improve LLM-based time series forecasting[^src-event-driven-ts-forecasting].

## Motivation

The basic event-driven reasoning pipeline in VoT operates in an unsupervised manner — the LLM generates predictions from summaries without explicit error feedback. This may introduce suboptimal guidance for numerical prediction patternsoro, potentially amplifying prediction errors[^src-event-driven-ts-forecasting]. HIC addresses this by providing error-informed guidance from historically corrected reasoning examples.

## Mechanism

### Knowledge Base Construction (Training Phase)

1. The Reasoner LLM generates initial predictions and reasoning processes from summaries
2. Using ground truth values, the Reasoner creates **corrected reasoning** — explanations of what caused prediction errors and how to derive correct values
3. Pairs of summary embeddings and their corresponding corrections are stored in a knowledge base: $\mathcal{K} = \{(\text{Embed}(\mathcal{S}_i), \mathcal{C}_i)\}_{i=1}^{M}$[^src-event-driven-ts-forecasting]

### Retrieval-Guided Prediction (Inference Phase)

1. For a new data pair, the LLM generates a summary
2. HIC retrieves the most similar historical example from the knowledge base using embedding similarity
3. The retrieved correction serves as an in-context example in the reasoning prompt, providing error-informed guidance[^src-event-driven-ts-forecasting]

## Key Properties

- **No fine-tuning required**: HIC achieves error-informed learning purely through retrieval and prompt engineering, making it efficient and scalable across domains[^src-event-driven-ts-forecasting]
- **Error-informed**: Retrieved examples contain analysis of previous prediction errors, helping the Reasoner understand and avoid similar mistakes
- **Similarity-based retrieval**: Uses embedding similarity to find the most relevant historical context for the current prediction

## Comparison with Related Techniques

| Technique | Model | Goal | Method |
|-----------|-------|------|--------|
| **HIC** | VoT | Error-informed LLM reasoning | Retrieve corrected historical examples |
| [[retrieval-augmented-statistical-channel|RAGC]] | [[channelmts|ChannelMTS]] | Retrieve similar channel statistics | Pre-cached高铁 map lookup |
| [[content-condenser-reconstruction|Content Condenser]] | [[mindts|MindTS]] | Filter redundant text | IB-based Bernoulli masking |

HIC is unique in using **corrected reasoning examples** (not raw data) as retrieval targets, providing explicit error analysis rather than just similar data points[^src-event-driven-ts-forecasting].

## Related Pages

- [[vot]] — the VoT model
- [[event-driven-reasoning]] — the reasoning paradigm HIC enhances
- [[multi-level-alignment]] — complementary alignment approachorate
- [[source-event-driven-ts-forecasting]] — source summary

[^src-event-driven-ts-forecasting]: [[source-event-driven-ts-forecasting]]
