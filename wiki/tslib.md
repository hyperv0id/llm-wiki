---
title: "TSLib"
type: entity
tags:
  - time-series
  - benchmark
  - survey
  - open-source
  - Tsinghua
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# TSLib

**TSLib** (Time Series Library) is an open-source benchmark for deep time series models, introduced by Wang, Wu, Dong, Liu, Wang, Long, and Wang (Tsinghua University, 2025). It implements 30 prominent models across 30 datasets and 5 analysis tasks—forecasting, classification, imputation, and anomaly detection—providing a standardized evaluation framework for the time series research community[^src-deep-time-series-survey].

## Overview

The deep time series field has been fragmented, with prior literature focusing either on specific architectures (Transformers, GNNs) or single tasks. TSLib fills this gap by providing a unified benchmark that enables fair, reproducible comparison across model families. The accompanying survey organizes models from two perspectives: basic modules (stationarization, decomposition, Fourier analysis) and backbone architectures (MLP-based, RNN-based, CNN-based, GNN-based, and Transformer-based)[^src-deep-time-series-survey].

## Key Innovation

TSLib's key contributions are twofold[^src-deep-time-series-survey]:

1. **Comprehensive Model Coverage** — Implements 30 models spanning five backbone families, including classic methods (ARIMA, N-BEATS), frequency-domain models (FreTS, FEDformer, FiLM), periodicity-aware architectures (TimesNet, Autoformer), and cutting-edge approaches (Mamba, Koopa, PatchTST, iTransformer, DLinear).

2. **Standardized Evaluation** — Evaluates 13 advanced models across 30 datasets and multiple tasks using consistent protocols, enabling meaningful cross-architecture comparisons. A key finding is that no single architecture dominates universally—model-task fit matters significantly.

## Architecture

TSLib is organized around a modular codebase where models share common preprocessing, evaluation, and logging infrastructure. The 30 implemented models are categorized by token granularity (point-wise, patch-wise, series-wise) and backbone family (MLP, RNN, CNN, GNN, Transformer)[^src-deep-time-series-survey].

## Performance

TSLib's evaluation of 13 advanced models across 30 datasets reveals that no single architecture dominates universally—model-task fit matters significantly. Patch-wise Transformers (PatchTST) excel at long-horizon forecasting, TimesNet's 2D reshaping proves effective across multiple tasks, and simpler architectures (DLinear, N-BEATS) remain surprisingly competitive, challenging the assumption that complexity equates to superiority[^src-deep-time-series-survey].

## Connections

- **[[timesnet]]** — TimesNet is one of the key models benchmarked in TSLib, with its 2D reshaping approach shown to be effective across multiple tasks.
- **[[autoformer]]** — Autoformer is included in TSLib as a foundational periodicity-based Transformer and serves as a baseline for evaluating newer frequency-domain and decomposition-based models.
- **[[fedformer]]** — FEDformer is benchmarked in TSLib as a representative frequency-domain Transformer, demonstrating the effectiveness of Fourier and wavelet attention.
- **[[source-timesnet|TimesNet (source)]]** — TimesNet's task-general design philosophy aligns with TSLib's goal of providing a comprehensive benchmarking ecosystem.

[^src-deep-time-series-survey]: [[source-deep-time-series-survey]]
