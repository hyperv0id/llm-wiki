---
title: "Source: LSTNet — Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks"
type: source-summary
tags:
  - time-series
  - multivariate
  - deep-learning
  - lstnet
  - cross-dimension
created: 2026-05-31
last_updated: 2026-05-31
source_count: 0
confidence: high
status: active
---

# Source: LSTNet

**Authors**: Guokun Lai, Wei-Cheng Chang, Yiming Yang, Hanxiao Liu (Carnegie Mellon University)
**Venue**: SIGIR 2018
**Citations**: ~1,728

## Core Thesis

LSTNet is the first deep learning framework to explicitly model both short-term and long-term temporal patterns in multivariate time series (MTS) forecasting. It combines three neural components and one linear component: CNN extracts cross-dimension local patterns, RNN (GRU) captures long-term trends, a novel Skip-RNN with periodic skip connections captures very-long-range seasonal patterns, and a parallel autoregressive (AR) linear model handles scale changes that neural networks are insensitive to.

## Key Contributions

1. **First cross-dimension deep MTS model** — before MTGNN, Crossformer, or any other cross-dimension approach, LSTNet used CNN to extract local dependency patterns *among variables* (not just over time).

2. **Skip-RNN for periodic patterns** — the recurrent-skip component adds skip connections between hidden cells p steps apart (e.g., p=24 for hourly data to capture daily cycle). This extends the temporal span of gradient flow beyond what standard GRU/LSTM can reach.

3. **AR component for scale robustness** — neural networks struggle with non-periodic scale changes (e.g., electricity consumption surges on holidays). A lightweight linear AR model running in parallel makes the prediction sensitive to input scale while the neural part handles complex patterns. Ablation shows removing AR causes the *largest performance drop across nearly all datasets*.

4. **Temporal Attention alternative** — when period p is unknown or dynamic, the LSTNet-Attn variant replaces Skip-RNN with a learned attention over hidden states.

## Method Summary

Input X ∈ R^(n×T) passes through: (1) CNN with n-height filters extracting cross-variable local patterns → (2) GRU recurrent layer + parallel Skip-RNN with skip length p → (3) dense fusion + (4) addition of AR linear prediction Ŷ = h^D + h^L.

Evaluated on Traffic, Solar-Energy, Electricity, Exchange-Rate at horizons 3/6/12/24. LSTNet-skip achieves 17 bold-face best results, LSTNet-Attn achieves 7. Dominant on data with clear periodicity; comparable to AR/LRidge on Exchange-Rate (no periodicity).

## Limitations

- Requires manual tuning of skip length p; no automatic period detection
- CNN treats all variable dimensions equally, ignoring attribute heterogeneity
- Not effective on datasets lacking periodic patterns (e.g., Exchange-Rate)
- AR coefficients shared across all dimensions, limiting per-variable flexibility
