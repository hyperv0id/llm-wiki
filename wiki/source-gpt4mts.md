---
title: "DP-GPT4MTS: Dual-Prompt LLM for Textual-Numerical Time Series Forecasting"
type: source-summary
tags:
  - llm
  - prompt-engineering
  - multimodal
  - time-series
  - 2025
created: 2026-07-07
last_updated: 2026-07-07
source_count: 1
confidence: medium
status: active
---

# DP-GPT4MTS: Dual-Prompt Large Language Model for Textual-Numerical Time Series Forecasting

Liu, Wang & Zhu (Dalian University of Technology, Guangzhou University, arXiv 2025) propose **DP-GPT4MTS**, a dual-prompt framework that extends LLM-based time series forecasting by explicitly separating task instruction (explicit prompt) from context-aware textual embedding (textual prompt)[^src-gpt4mts].

## Dual-Prompt Architecture

Existing single-prompt LLM methods (GPT4MTS, TimeLLM) struggle to handle multimodal textual-numerical time series data because a single prompt conflates task-level instruction with instance-level text context, introducing redundancy. DP-GPT4MTS addresses this with two complementary prompts:

**1. Explicit Prompt (hard prompt):** A fixed prefix prepended to the input sequence, containing task instructions ("Predict the next T steps..."), input statistics (min, max, median, trend direction, top lag values), and a natural language description noting that each time point has associated events/news. This is tokenized by the frozen GPT-2 tokenizer into embeddings[^src-gpt4mts].

**2. Textual Prompt (soft prompt):** Timestamped text summaries (e.g., GDELT event descriptions) are encoded via a frozen BERT model. The CLS token embeddings are extracted per time step, then refined through a multi-head self-attention mechanism and linear projection to produce context-aware soft prompt embeddings. These are injected alongside the explicit prompt and time series embeddings[^src-gpt4mts].

Time series inputs are processed with RevIN (reversible instance normalization) and patching before concatenation with prompt embeddings into the frozen GPT-2 backbone. Position embeddings and layer normalization layers are fine-tuned while the rest of GPT-2 remains frozen.

## Datasets and Empirical Results

Experiments use two multimodal time series sources:
- **GDELT-based dataset** (Jia et al., 2024): 10 event types across 53 US regions, daily frequency, lookback=15, predict=7 days
- **Time-MMD dataset** (Liu et al., 2024a): agriculture (monthly, lookback=12, predict=4 months) and public health (weekly, lookback=36, predict=12 weeks)

DP-GPT4MTS achieves the lowest MSE on 8/10 GDELT events and best average MSE (0.976 vs GPT4MTS 0.997) and MAE. On Time-MMD, it achieves MSE/MAE of 0.098/0.211 (Agriculture) and 0.890/0.601 (Public Health), outperforming PatchTST, Autoformer, iTransformer, TimeLLM, and GPT4MTS[^src-gpt4mts].

## Significance

DP-GPT4MTS provides a clean architectural insight: decoupling task-level instruction from instance-level text context via dual prompts improves LLM-based multimodal time series forecasting. The GPT-2 backbone remains mostly frozen (only position/layer-norm fine-tuned), making the approach parameter-efficient. The GDELT-based dataset with 10 event categories offers a standardized benchmark for textual-numerical time series research[^src-gpt4mts].

## 相关页面

- [[source-exollm]] — LLM 外生变量预测

[^src-gpt4mts]: [[source-gpt4mts]] — DP-GPT4MTS: Dual-Prompt Large Language Model for Textual-Numerical Time Series Forecasting (Liu, Wang & Zhu, arXiv 2025)
