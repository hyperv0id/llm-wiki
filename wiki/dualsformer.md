---
title: "Dualformer"
type: entity
tags:
  - time-series
  - forecasting
  - frequency-domain
  - transformer
  - dual-domain
created: 2026-04-28
last_updated: 2026-04-29
source_count: 1
confidence: medium
status: active
---

# Dualformer

**Dualformer** is a dual-domain Transformer framework for long-term time series forecasting (LTSF) proposed by Jingjing Bai and Yoshinobu Kawahara (University of Osaka & RIKEN AIP) in January 2026. It addresses the inherent low-pass filtering effect of standard Transformer self-attention by introducing structured frequency allocation across network depth[^src-dualformer].

## Overview

Standard Transformers suffer from undifferentiated frequency propagation across layers, progressively suppressing high-frequency signals. Dualformer addresses this by allocating distinct frequency bands to different network depths[^src-dualformer].

## Key Innovation

Dualformer introduces three core innovations[^src-dualformer]:

1. **Dual-Branch Architecture** — A time branch applies vanilla self-attention after converting selected frequency components via inverse FFT, capturing local temporal dependencies. A frequency branch uses autocorrelation-based attention in the frequency domain (inspired by Autoformer's Wiener-Khinchin theorem approach), modeling global periodic patterns.

2. **Hierarchical Frequency Sampling (HFS)** — Decomposes the frequency spectrum across model depth. Lower layers receive high-frequency components (localized, fast-varying patterns), while deeper layers shift to low-frequency components (long-term trends). Sampling intervals are dynamically adjusted with overlap control to prevent information gaps.

3. **Periodicity-Aware Weighting** — Computes the harmonic energy ratio of each input sequence. A theoretical lower bound (Theorem 1) justifies using this ratio to determine the soft weight between the frequency and time branches. Strongly periodic signals favor the frequency branch; aperiodic signals favor the time branch.

## Architecture

Inputs are processed through parallel time-domain and frequency-domain pathways. The time branch uses IFFT followed by vanilla self-attention. The frequency branch applies autocorrelation-based attention directly on frequency representations. HFS dynamically assigns frequency bands per layer, structurally preventing low-frequency dominance. The periodicity-aware weighting module fuses both branches based on input characteristics[^src-dualformer].

## Performance

Dualformer achieves top rank in 13 of 16 average cases and 44 of 64 multivariate outcomes across eight benchmarks (ETTh1/ETTh2, ETTm1/ETTm2, Electricity, Solar, Traffic, Weather) with prediction lengths 96–720. Ablations confirm the frequency branch is most critical, and HFS outperforms fixed frequency selection. Performance on strongly periodic Traffic is weaker, suggesting dedicated period-extraction methods retain an advantage on highly seasonal data[^src-dualformer].

## Connections

- **[[fedformer]]** — FEDformer also operates in the frequency domain but uses fixed random mode selection rather than input-adaptive hierarchical sampling. Dualformer's periodicity-aware weighting directly addresses this limitation.
- **[[autoformer]]** — Autoformer's autocorrelation mechanism (Wiener-Khinchin theorem) directly inspired Dualformer's frequency-branch attention design.
- **HyperD** — Both frameworks employ structured frequency allocation, though HyperD focuses on decoupling short-term vs. long-term periodicity for traffic forecasting, while Dualformer uses layer-wise hierarchical frequency sampling for general LTSF.

[^src-dualformer]: [[source-dualformer]]
