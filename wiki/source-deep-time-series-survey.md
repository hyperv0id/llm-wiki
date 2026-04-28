---
title: "Deep Time Series Models: A Comprehensive Survey and Benchmark"
type: source-summary
tags:
  - time-series
  - forecasting
  - survey
  - benchmark
created: 2026-04-28
last_updated: 2026-04-28
source_count: 0
confidence: high
status: active
---

# Deep Time Series Models: A Comprehensive Survey and Benchmark

## Overview

This survey by Wang, Wu, Dong, Liu, Wang, Long, and Wang (Tsinghua University, 2025) provides a comprehensive review of deep learning models for time series analysis, covering both basic modules and full model architectures. It introduces the **Time Series Library (TSLib)**, an open-source benchmark implementing 30 prominent models across 30 datasets and 5 analysis tasks: forecasting, classification, imputation, and anomaly detection. The survey fills a gap in existing literature, which previously focused either on specific architectures (Transformers, GNNs) or single tasks.

## Key Method

The paper organizes deep time series models from two perspectives:

1. **Basic Modules** — foundational techniques integrated into modern models:
   - **Stationarization** (RevIN, Non-Stationary Transformer, SAN) — normalizing non-stationary inputs
   - **Decomposition** — seasonal-trend decomposition (Autoformer), basis expansion (N-BEATS, N-HiTS), matrix factorization (TRMF, DeepGLO)
   - **Fourier Analysis** — time-domain modeling (TimesNet, FEDformer, FiLM, FITS) and frequency-domain modeling (FreTS, StemGNN, TSLANet)

2. **Model Architectures** — five backbone families:
   - **MLP-based** (N-BEATS, DLinear, TSMixer, TimeMixer, Koopa)
   - **RNN-based** (LSTNet, DeepAR, SSMs like Mamba, Neural ODEs)
   - **CNN-based** (SCINet, TCN, MICN, ModernTCN, TimesNet's 2D reshaping)
   - **GNN-based** (DCRNN, STGCN, Graph WaveNet, AGCRN, MTGNN, StemGNN)
   - **Transformer-based** — categorized by token granularity: point-wise (Informer, LogSparse, Pyraformer, Autoformer), patch-wise (PatchTST, Crossformer), and series-wise (iTransformer)

The TSLib benchmark evaluates 13 advanced models across forecasting, classification, imputation, and anomaly detection tasks. A key finding is that **models with specific structures are well-suited for distinct analytical tasks** — no single architecture dominates universally.

## Results

- TSLib enables fair, reproducible comparison across 30 models and 30 datasets
- Empirical results confirm that model-task fit matters: e.g., patch-wise Transformers (PatchTST) excel at long-horizon forecasting, while TimesNet's 2D reshaping is effective across multiple tasks
- The paper identifies that simpler architectures (DLinear, N-BEATS) can be surprisingly competitive, challenging the assumption that more complex models are always better

## Critique

- **Strengths**: The two-level organizational scheme (modules → architectures) is pedagogically sound. TSLib fills a genuine need for standardized benchmarking in a fragmented field. Coverage of both classic (ARIMA, N-BEATS) and cutting-edge (Mamba, Koopa, TimesNet) methods is thorough.
- **Limitations**: The survey is model-centric rather than problem-centric, so practitioners may find it hard to navigate by task. The TSLib benchmark, while comprehensive, focuses only on 13 of 30 implemented models for the full evaluation. The treatment of [[traffic-forecasting|spatio-temporal forecasting]] is brief, treated mainly as a subset of multivariate time series rather than a distinct domain with its own challenges (e.g., graph structure, dynamic correlations).
- **Relevance to this wiki**: The survey's taxonomy directly contextualizes the models covered in this wiki — [[hyperd|HyperD]] (Transformer/periodicity-decoupling), ST-ResNet (CNN/residual), and ASTGCN (GCN/attention) — within the broader deep time series landscape.
