---
title: "NuwaTS — a Foundation Model Mending Every Incomplete Time Series"
type: source-summary
tags:
  - time-series
  - data-imputation
  - foundation-model
  - pretrained-language-model
  - contrastive-learning
  - prefix-tuning
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

**NuwaTS** is a foundation model for general incomplete time series imputation proposed by Jinguo Cheng, Chunwei Yang, Wanlin Cai (Sichuan University), Yuxuan Liang (HKUST-Guangzhou), Yuankai Wu (Sichuan University, corresponding), and Qingsong Wen (Squirrel Ai Learning), arXiv:2405.15317v3 (2 Oct 2024)[^src-nuwats]. It repurposes the first six layers of a Pre-trained Language Model (PLM, GPT-2 by default) into a single "one-for-all" model that can mend missing values in any incomplete time series regardless of domain, variable, or missing pattern[^src-nuwats]. Code: github.com/Chengyui/NuwaTS.

## Problem

Existing imputation models require specialized designs tailored to a specific missing pattern, variable set, or domain, and are evaluated with time-wise train/validation/test splits that test only future observations of the *same* variables — failing to assess cross-variable and cross-domain generalization[^src-nuwats].

## Core Contributions

1. **Variable-wise benchmarking protocol** — partitions multivariate data along the variable (sensor) dimension in a 1:1:1 ratio, so a model trains, validates, and tests on *disjoint* variables, simulating deployment on entirely new sensors/domains[^src-nuwats]. See [[variable-wise-partitioning]].

2. **NuwaTS model** — patches each series and adds three specialized embeddings: a **statistical embedding** (min/median/max/trend, at both series and patch level), a **missing embedding** (a learnable vector scaled by each patch's mask ratio), and a **domain-specific embedding** prefix. A **contrastive learning** module (InfoNCE + MSE) feeds two differently-masked views of each input through the PLM and pulls same-patch representations together, making embeddings mask-invariant[^src-nuwats].

3. **Plug-and-play prefix fine-tuning** — a P-tuning-v2-style domain-specific prefix ([Keyₚ,Valueₚ] = P + βK̂, β=0.01) injected into every PLM layer with the backbone frozen; the prefix needs <100 KB vs 331.77 MB for the full GPT-2 model and reverts to the one-for-all model when removed[^src-nuwats]. See [[plug-and-play-prefix-tuning]].

## Four Versions & Results

Four GPT-2-based variants are studied: specific (single domain), one-for-all (17.6M fused samples from ETT/Weather/ECL/PEMS), fine-tuned, and cross-domain (pre-trained only on LargeST's 100.1M samples)[^src-nuwats]. Across 10 datasets and 9 missing rates (0.1–0.9), one-for-all NuwaTS beats domain-specific SOTA (SAITS, BRITS, TimesNet, PatchTST, GPT4TS) at nearly all rates; training on the fused multi-domain set further improves generalization, evidencing a **scaling law** for imputation[^src-nuwats]. As a channel-independent method it zero-shots to datasets with different variable counts (LargeST⇒ECL/Weather), beats PatchTST zero-shot, is robust under continuous missing, and on ETT reaches 100%-data quality with only 10% fine-tuning data[^src-nuwats]. NuwaTS-imputed data also improves downstream TimesNet forecasting, and the model converts to a forecaster by appending masked padding tokens[^src-nuwats].

## Key Findings (Ablations)

Statistical, missing, and contrastive components are all necessary; freezing the backbone hurts badly; **training from scratch without NLP weights sharply degrades zero-shot transfer** (cross-modality pre-training is meaningful)[^src-nuwats]. Data quantity matters more than the specialized inductive biases (gains are larger on the small ETTh1 than the huge LargeST). GPT-2 > BERT > LLaMA2 as backbone, and **simple linear patch embedding beats Time-LLM-style text-alignment** for incomplete series[^src-nuwats].

## Limitations

Trained on fixed length-96 segments; longer segments or entirely-missing segments may require further fine-tuning[^src-nuwats].

[^src-nuwats]: [[source-nuwats]]
