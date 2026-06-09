---
title: "source: FSTLLM — Spatio-Temporal LLM for Few Shot Time Series Forecasting (ICML 2025)"
type: source-summary
tags:
  - source-summary
  - llm
  - spatio-temporal
  - few-shot-learning
  - traffic-forecasting
  - icml-2025
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# FSTLLM: Spatio-Temporal LLM for Few Shot Time Series Forecasting

**FSTLLM** (Jiang, Chen, Li, Chao, Liu & Cong, ICML 2025, PMLR 267) is a flexible, LLM-augmented framework for **few-shot** multivariate time series forecasting that plugs the contextual reasoning of large language models into existing spatio-temporal forecasting models[^src-fstllm].

## Motivation

State-of-the-art forecasters — Spatio-Temporal Graph Neural Networks ([[dcrnn|STGNNs]]) and Time Series Foundation Models (TSFMs like GPT4TS, [[time-llm|Time-LLM]]) — require large training corpora and degrade sharply when data is scarce, which is common in practice (collecting months of sensor data is costly)[^src-fstllm]. Two specific gaps motivate FSTLLM: (1) these models normalize series into numerical vectors and ignore real-world contextual knowledge (geography, urban context, human behavior); (2) prior LLM-for-TS methods either fine-tune on purely numerical inputs or prepend generic task instructions, underusing the LLM's reasoning. FSTLLM also notes the critique (Tan et al., 2024) that some LLM-TS methods' LLM component can be replaced by a simple layer without loss — and argues richer contextual prompting is the remedy[^src-fstllm].

## Method

FSTLLM has three modules[^src-fstllm]:

1. **LLM-Enhanced Graph Construction** — node-specific textual documents (e.g., parking-lot descriptions, user reviews) are encoded by a frozen LLaMA-2-7B; final-layer hidden states are projected (FFN) into node embeddings, passed through a graph-attention layer, and normalized with the **α-Entmax** sparse activation (α=2.0) to yield a semantically meaningful adjacency matrix A.
2. **STGNN backbone** — A feeds a graph-diffusion-convolution GRU (GTS-style) that produces numerical prediction tokens C. The backbone is swappable.
3. **Domain Knowledge Injection** — a LLaMA-2-7B is SFT-fine-tuned via QLoRA on prompts with six components (task instruction, node description, node pattern, historical input, numerical prediction token, future token) to calibrate the STGNN's numerical tokens into context-aware, human-like predictions.

## Contributions & Results

- Plug-and-play: FSTLLM augments existing forecasters (GPT4TS, iTransformer) by substituting their predictions for the STGNN tokens, **without retraining** those models[^src-fstllm].
- On two few-shot real-world datasets — **Nottingham** (19 car-park availability) and **ECL** (19 electricity clients) — FSTLLM wins 22/24 (Nottingham) and 25/36 (ECL) evaluations, with ~30% MAPE reduction (Nottingham) and >50% MAPE reduction vs GPT4TS/iTransformer (ECL)[^src-fstllm].
- Data efficiency: FSTLLM trained on **3 days** beats baselines trained on **30 days** (10× more data)[^src-fstllm].
- Ablation: removing Domain Knowledge Injection causes the largest degradation; replacing LLM graph construction with cosine-similarity also hurts[^src-fstllm].
- Interpretability: FSTLLM produces textual rationales (e.g., capping parking predictions at the lot's 512-space capacity)[^src-fstllm].

## Limitations

LLaMA-2-7B inference is slow (single-GPU, no data parallelism), so ECL was restricted to 19 of 320 clients[^src-fstllm]. Both datasets are small (≤19 nodes); larger-scale validation is left to future work[^src-fstllm].

## Links

- [[fstllm]] — method/entity page
- [[few-shot-traffic-forecasting]] — task framing (proposed)
- [[time-llm]] · [[urbangpt]] · [[gpd]] — related LLM/few-shot spatio-temporal methods
- [[traffic-forecasting]] · [[mtgnn]] · [[itransformer]] — backbones and domain

[^src-fstllm]: [[source-fstllm]]
