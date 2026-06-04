---
title: "source-streasoner"
type: source-summary
tags:
  - spatio-temporal
  - reasoning
  - llm
  - time-series
  - reinforcement-learning
  - benchmark
created: 2026-06-04
last_updated: 2026-06-04
source_count: 0
confidence: high
status: active
---

# STReasoner: Empowering LLMs for Spatio-Temporal Reasoning in Time Series via Spatial-Aware Reinforcement Learning

Ni et al. (Emory University, Microsoft, Griffith University; arXiv 2026) propose **STReasoner**, the first TS-LM designed for explicit spatio-temporal reasoning over time series data, along with **ST-Bench** for evaluation and **S-GRPO** for training[^src-streasoner]. Source: `raw/streasoner-ni-2026.pdf`.

## Problem: Spatio-Temporal Reasoning

Spatio-temporal reasoning in time series requires linking observations across multiple nodes and time steps through spatial dependencies and temporal dynamics — answering queries like "Which source node caused the congestion at Node 2 at 9:00?" Existing time series LMs focus on prediction accuracy rather than explicit reasoning, and lack spatial dependency modeling[^src-streasoner].

## Key Contributions

1. **ST-Bench**: A four-task benchmark (etiological reasoning, entity identification, correlation reasoning, in-context forecasting) built via a network SDE-based multi-agent data synthesis pipeline that generates spatio-temporal data with controllable dynamics and aligned textual descriptions[^src-streasoner].

2. **STReasoner**: A unified spatio-temporal reasoning model that uses an MLP-based time series encoder (patchify → 5-layer MLP) to embed numerical time series, interleaves time series tokens with text tokens (including graph structure as text), and processes through an LLM backbone[^src-streasoner].

3. **S-GRPO** (Spatial-Aware Group Relative Policy Optimization): A RL algorithm that computes contrastive rewards — comparing model performance with vs. without spatial structure — to explicitly incentivize spatial reasoning. Only when spatial information improves answer accuracy does the model receive bonus reward[^src-streasoner].

## Training Paradigm

Three stages[^src-streasoner]:
1. **Alignment pretraining** (ST-Align, 153.7K QA pairs): Bridge TS and text modalities via basic temporal/spatial/ST questions
2. **SFT** (ST-CoT, via rejection sampling with Claude-4.5-Sonnet): Inject spatio-temporal reasoning priors through CoT annotations
3. **S-GRPO**: Reinforce spatial-aware reasoning via contrastive spatial reward

## Key Results

| Task | STReasoner-8B | GPT-5.2 (text) | GPT-5.2 (img) |
|------|:--:|:--:|:--:|
| Etiological | **95.65%** | 83.09% | 86.47% |
| Entity | **75.71%** | 38.78% | 40.54% |
| Correlation | **87.12%** | 58.79% | 65.08% |
| Forecasting (MAE) | 65.59 | 63.99 | 64.70 |

STReasoner achieves these gains at **0.004×** the cost of proprietary models (GPT-5.2: $22.48 vs STReasoner: $0.27)[^src-streasoner]. Zero-shot on real-world CausalRivers: **98.82%** (GPT-5.2: 22.32%, Claude: 83.18%). Ablation: full Align+SFT+S-GRPO critical; removing S-GRPO drops etiological ACC from 95.65% to 91.79%[^src-streasoner].

## Limitations

- Relies on synthetic data; broader real-world datasets needed
- Simple MLP encoder may be insufficient for complex real-world scenarios
- Training cost: ~216 GPU-hours total (Align: 107h, SFT: 65h, RL: 43h)[^src-streasoner]

[^src-streasoner]: [[source-streasoner]]
