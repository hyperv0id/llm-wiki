---
title: "TimesNet"
type: entity
tags:
  - time-series
  - forecasting
  - periodicity
  - foundation-model
  - ICLR-2023
created: 2026-04-28
last_updated: 2026-08-29
source_count: 4
confidence: medium
status: active
---

# TimesNet

**TimesNet** is a task-general foundation model for time series analysis proposed by Haixu Wu et al. (Tsinghua University), published at ICLR 2023. It transforms 1D time series into 2D tensors based on multi-periodicity, enabling the application of 2D convolutional backbones from computer vision to time series analysis. TimesNet achieves consistent state-of-the-art across five mainstream tasks: short- and long-term forecasting, imputation, classification, and anomaly detection[^src-timesnet].

## Overview

Time series exhibit intraperiod-variation (within a single period) and interperiod-variation (across the same phase of different periods). Reshaping 1D series into 2D tensors maps these to columns and rows, making them amenable to 2D convolutional kernels from vision. Unlike prior methods, TimesNet taps into the mature vision backbone ecosystem[^src-timesnet].

## Key Innovation

TimesNet's core is the **TimesBlock**, with four steps[^src-timesnet]:

1. **Period Discovery** — FFT identifies the top-k significant frequencies and periods from the input series.
2. **1D-to-2D Reshape** — For each discovered period, the 1D series is reshaped into a 2D tensor where columns are intraperiod steps and rows are interperiod steps.
3. **2D Representation Learning** — A parameter-efficient Inception block (multi-scale 2D convolutions) processes each tensor, capturing both variation types.
4. **Adaptive Aggregation** — The k representations are fused via softmax-weighted sum based on FFT amplitude values.

A key advantage is generality: any vision architecture (ResNet, ConvNeXt, etc.) can replace the Inception block, bridging time series analysis with the broader computer vision community.

## Architecture

TimesNet stacks multiple TimesBlocks in sequence. Each block independently discovers periods from its input, allowing layers to capture periodicity at varying abstraction levels. Adaptive aggregation ensures periods with stronger spectral energy contribute more to the output, making TimesNet a flexible backbone across tasks without task-specific modifications[^src-timesnet].

## Performance

TimesNet outperforms 15+ baselines (Autoformer, FEDformer, DLinear, Informer, etc.) on long-term forecasting across ETT, Electricity, Traffic, Weather, Exchange, and ILI. On M4 short-term forecasting, it surpasses specialized models like N-BEATS and N-HiTS. For classification, it achieves 73.6% accuracy on UEA, surpassing Rocket and Flowformer, and achieves best F1 scores across anomaly detection benchmarks (SMD, MSL, SMAP, SWaT, PSM)[^src-timesnet].

## 综述归类

Wang & Du 等人的 MTSI 综述将 TimesNet 归为预测式-CNN 类插补方法（Table 1 缺失机制标注 MCAR），归因于其用 Fast Fourier Transform 将 1D 序列重组为 2D 格式、从而适配 CNN 处理的思路[^src-mts-imputation-survey]。这是综述从插补任务视角给出的单一任务归类；TimesNet 本身是覆盖五类任务的通用架构（见本页 Overview），两套口径分立。

## Connections

- **[[informer]]** — Informer (AAAI 2021 Best Paper) pioneered efficient Transformer-based LSTF and is listed as a key baseline in TimesNet's experiments[^src-timesnet].
- **[[autoformer]]** — Autoformer's progressive decomposition and periodicity analysis via autocorrelation precede TimesNet's approach. TimesNet extends the idea by explicitly modeling multiple periods through 2D reshaping[^src-timesnet].
- **[[source-autoformer|Autoformer (source)]]** — Autoformer is extensively cited by TimesNet as a foundational reference for periodicity-based time series modeling[^src-timesnet].
- **[[source-hyperd-hybrid-periodicity-decoupling|HyperD]]** — HyperD decouples short-term and long-term periodicity into separate specialized pathways, an alternative design to TimesNet's shared 2D convolution approach over multiple discovered periods[^src-hyperd-hybrid-periodicity-decoupling]
- **[[unica|UniCA]]** — 作为时间序列基础模型（TSFM）之一，TimesFM 可作为 UniCA 的适配基线。UniCA 框架可为其添加异构协变量支持，同时保持 TimesNet/TimesFM 主干冻结[^src-unica]

## 相关页面

- [[unified-covariate-adaptation]] — 统一协变量适应概念
- [[covariate-homogenization]] — 协变量同质化技术
- [[covariate-fusion-module]] — 协变量融合模块技术
- [[mts-imputation-taxonomy]] — MTSI 综述的分类框架，TimesNet 归为预测式-CNN 类

## 引用

[^src-unica]: [[source-unica]]
[^src-timesnet]: [[source-timesnet]]
[^src-hyperd-hybrid-periodicity-decoupling]: [[source-hyperd-hybrid-periodicity-decoupling]]
[^src-mts-imputation-survey]: [[source-mts-imputation-survey]]
