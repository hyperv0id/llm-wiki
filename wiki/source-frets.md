---
title: "Frequency-domain MLPs are More Effective Learners in Time Series Forecasting"
type: source-summary
tags:
  - time-series
  - forecasting
  - frequency-domain
  - mlp
  - energy-compaction
  - global-view
created: 2026-04-28
last_updated: 2026-04-28
source_count: 0
confidence: high
status: active
---

# FreTS

**FreTS** (Frequency-domain MLPs for Time Series forecasting) is a simple yet effective architecture proposed by Kun Yi et al. (Beijing Institute of Technology, University of Oxford, et al.), published at NeurIPS 2023. It applies multi-layer perceptrons (MLPs) in the frequency domain rather than the time domain, leveraging two inherent advantages: global view and energy compaction.

## Overview

MLP-based forecasting methods (e.g., DLinear, N-BEATS) suffer from point-wise mappings that cannot handle global dependencies and suffer from information bottlenecks due to volatile local momenta[^src-frets]. FreTS addresses this by exploring a novel direction: applying MLPs directly on frequency-domain representations. Theoretical analysis shows that frequency-domain MLPs are equivalent to global convolutions in the time domain (Theorem 2), enabling them to capture complete signal views. Additionally, the energy compaction property of Fourier transforms (Theorem 1, Parseval's theorem) means most signal energy concentrates in a few frequency components, allowing frequency-domain MLPs to focus on key patterns while filtering noise.

## Key Method

FreTS has two main stages[^src-frets]:

1. **Domain Conversion**: The input time series is transformed into frequency-domain complex numbers via Discrete Fourier Transform (DFT). The spectrum consists of real and imaginary parts representing different frequency components.

2. **Frequency Learning (FreMLP)**: Redesigned MLPs for complex numbers. The real and imaginary parts are processed by two separate MLPs, with the output stacked to form complex-valued results. This is formalized as: Y = σ(Re(Y)W_r − Im(Y)W_i + B_r) + j·σ(Re(Y)W_i + Im(Y)W_r + B_i).

FreTS applies these stages at two scales:
- **Frequency Channel Learner**: Operates along the channel dimension (per timestamp, shared weights across timestamps), modeling inter-series dependencies.
- **Frequency Temporal Learner**: Operates along the time dimension (per channel, shared weights across channels), modeling temporal dependencies.

The two learners are stacked sequentially with a final feed-forward projection layer producing the forecast.

## Results

Extensive experiments on 13 real-world benchmarks (7 short-term, 6 long-term) demonstrated FreTS's consistent superiority[^src-frets]. For short-term forecasting, FreTS outperformed 13 baselines including VAR, LSTNet, TCN, StemGNN, MTGNN, and AGCRN across all datasets (Solar, Wiki, Traffic, ECG, Electricity, COVID-19), with average improvements of 9.4% MAE and 11.6% RMSE. For long-term forecasting, FreTS outperformed Informer, Autoformer, FEDformer, PatchTST, and LTSF-Linear across all datasets (Weather, Exchange, Traffic, Electricity, ETTh1, ETTm1), with over 20% average reduction in MAE and RMSE compared to Transformer-based models. Ablation studies showed FreMLP improved DLinear and NLinear by 6.4% MAE and 11.4% RMSE on average. Efficiency analysis showed FreTS reduces parameters by at least 3× and training time by 3-10× compared to Transformer-based methods, with complexity O(N log N + L log L).

## Critique

FreTS's primary contribution is demonstrating that simple MLPs, when moved to the frequency domain, can match or outperform sophisticated Transformer architectures—a finding with significant practical implications for deployment-constrained settings. The dual-learner design (channel + temporal) is elegant and the theoretical grounding (global convolution equivalence, energy compaction) is rigorous. However, FreTS does not adaptively select frequency components—it uses all frequencies uniformly, unlike FEDformer's random selection or Dualformer's hierarchical sampling. The fixed architecture may struggle on datasets where different frequency bands carry vastly different predictive value. Performance on the Electricity and Traffic datasets, while still strong, shows less dramatic gains than on Weather and Exchange. The paper also does not explore very long lookback windows beyond 96 steps, leaving open questions about scalability.

[^src-frets]: [[source-frets]]
