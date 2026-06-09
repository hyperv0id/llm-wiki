---
title: "Few-Shot Spatio-Temporal / Traffic Forecasting"
type: concept
tags:
  - few-shot-learning
  - spatio-temporal
  - traffic-forecasting
  - data-scarce
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Few-Shot Spatio-Temporal / Traffic Forecasting

**Few-shot spatio-temporal forecasting** is the problem of predicting future traffic/urban states when only a small amount of historical data is available for the target setting (e.g., a few days rather than months)[^src-fstllm]. State-of-the-art [[dcrnn|STGNNs]] and Time Series Foundation Models are data-hungry: they require large corpora to learn complex spatial-temporal correlations and degrade sharply under data scarcity[^src-fstllm]. Because collecting large-scale sensor data is time-consuming and resource-intensive (potentially months), the data-scarce regime is common in real deployments[^src-fstllm].

## Why standard models struggle

With limited training signal, purely data-driven adaptive graph learners produce unstable node embeddings, and models that normalize series into numerical vectors cannot recover from missing context — they ignore real-world factors such as geography, urban context, and human behavioral patterns that are often decisive for accuracy[^src-fstllm].

## Solution families

- **Contextual-knowledge injection** — [[fstllm|FSTLLM]] (ICML 2025) leverages a large language model's common-sense reasoning, encoding node-specific text into a semantically meaningful adjacency matrix and fine-tuning the LLM to calibrate an STGNN's predictions; it reports that 3 days of data can outperform baselines trained on 30 days (10× more)[^src-fstllm].
- **Parameter-space transfer** — [[gpd|GPD]] (ICLR 2024) pre-trains a diffusion hypernetwork on source-city model parameters and generates a target-city forecaster from a few days of data (no LLM).
- **Foundation-model generalization** — [[unist|UniST]] and [[urbanpg|UrbanPG]] pursue zero-/few-shot generalization via masked pre-training and prompt tuning.

## Evaluation

A few-shot setup typically restricts the training split to the most recent short window (e.g., one week or three days) of the data, while keeping standard validation/test splits[^src-fstllm]. FSTLLM evaluates on the Nottingham car-park (19 nodes) and ECL electricity (19 clients) datasets, reporting MAE/RMSE/MAPE across horizons[^src-fstllm].

## Related

- [[traffic-forecasting]] — full-data spatio-temporal forecasting and its method families
- [[time-llm]] · [[urbangpt]] — LLM-based time series / spatio-temporal methods
- [[gpd]] — diffusion-based few-shot spatio-temporal transfer

[^src-fstllm]: [[source-fstllm]]

