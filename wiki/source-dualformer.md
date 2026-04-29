---
title: "Dualformer: Time-Frequency Dual Domain Learning for Long-term Time Series Forecasting"
type: source-summary
tags:
  - time-series
  - forecasting
  - frequency-domain
  - transformer
  - dual-domain
  - hierarchical-frequency-sampling
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# Dualformer

**Dualformer** is a dual-domain Transformer framework for long-term time series forecasting (LTSF), proposed by Jingjing Bai and Yoshinobu Kawahara (University of Osaka & RIKEN AIP) in January 2026. It addresses the inherent low-pass filtering effect of Transformer self-attention, which progressively attenuates high-frequency information critical for capturing fine-grained temporal variations.

## Overview

Standard Transformer-based models suffer from an undifferentiated propagation of frequency components across layers, causing high-frequency signals—representing rapid changes and short-term fluctuations—to be progressively suppressed as the network deepens[^src-dualformer]. This low-pass bias limits their effectiveness on heterogeneous or weakly periodic data. Dualformer rethinks frequency modeling from a layer-wise perspective, introducing a structured approach to allocate distinct frequency bands to different network depths.

## Key Method

Dualformer's architecture consists of three core components[^src-dualformer]:

1. **Dual-Branch Architecture**: A time branch and a frequency branch operate in parallel. The time branch applies vanilla self-attention after converting selected frequency components back to the time domain via inverse FFT (IFFT), capturing local temporal dependencies. The frequency branch uses an autocorrelation-based attention directly in the frequency domain (inspired by Autoformer's Wiener-Khinchin theorem approach) to model global periodic patterns.

2. **Hierarchical Frequency Sampling (HFS)**: This module explicitly decomposes the frequency spectrum across model depth. Lower layers receive high-frequency components (capturing localized, fast-varying patterns), while deeper layers shift attention to low-frequency components (modeling long-term trends). The sampling intervals are dynamically adjusted based on layer index, with overlap control to prevent information gaps. This structurally prevents low-frequency dominance by constraining what each layer can attend to.

3. **Periodicity-Aware Weighting**: A dynamic mechanism that computes the harmonic energy ratio of each input sequence—the proportion of total spectral energy concentrated in harmonic components. A theoretical lower bound is derived (Theorem 1) to justify using this ratio as a periodicity surrogate. The ratio determines the soft weight assigned to the frequency branch vs. the time branch: strongly periodic signals weight the frequency branch more heavily, while aperiodic signals favor the time branch.

## Results

Dualformer was evaluated on eight benchmark datasets (ETTh1/ETTh2, ETTm1/ETTm2, Electricity, Solar Energy, Traffic, Weather) with prediction lengths {96, 192, 336, 720}[^src-dualformer]. It achieved the top rank in 13 of 16 average cases and 44 of 64 total multivariate outcomes. Ablation studies confirmed the frequency branch is the most critical component (largest performance drop when removed), and HFS consistently outperformed fixed frequency selection strategies (low-frequency-only, high-frequency-only, full FFT, random band) on both strongly periodic (Traffic) and weakly periodic (Weather) datasets. Performance on the Traffic dataset was relatively weaker, likely because models like PDF that explicitly extract dominant periods have an advantage on highly seasonal data.

## Critique

Dualformer's strength lies in its principled, theoretically grounded approach to frequency allocation across layers—moving beyond uniform frequency processing. The periodicity-aware weighting is supported by a derived lower bound, providing more rigor than heuristic fusion methods. However, the hierarchical frequency sampling introduces an additional hyperparameter (the sampling ratio α), and the dual-branch architecture increases computational overhead compared to single-branch models. The weaker performance on highly periodic datasets (Traffic) suggests the model may sometimes fail to exploit strong periodic structures as effectively as dedicated period-extraction methods. The paper also focuses on fixed lookback windows of 96 steps, leaving open the question of sensitivity to varying input lengths. Future work directions include extensions to high-dimensional multivariate, irregularly sampled, and event-driven time series.

[^src-dualformer]: [[source-dualformer]]
