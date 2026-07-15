---
title: "EAC: Expand and Compress — Exploring Tuning Principles for Continual Spatio-Temporal Graph Forecasting"
type: source-summary
tags:
  - continual-learning
  - spatio-temporal
  - prompt-learning
  - traffic-forecasting
  - air-quality
  - wind-energy
  - iclr-2025
  - stgnn
created: 2026-07-18
last_updated: 2026-07-18
source_count: 0
confidence: high
status: active
---

# EAC: Expand and Compress

**Source**: Wei Chen, Yuxuan Liang. *Expand and Compress: Exploring Tuning Principles for Continual Spatio-Temporal Graph Forecasting.* ICLR 2025. [[eac-expand-and-compress-continual-stg-forecasting-iclr2025.pdf|PDF]]

## Core Contribution

EAC proposes a prompt-based continual spatio-temporal graph forecasting framework that freezes a backbone STGNN and adapts to streaming data solely through a lightweight, dynamically expandable **continuous prompt parameter pool**. The paper derives two fundamental tuning principles from empirical observation and theoretical analysis: **expand** (heterogeneity-guided prompt pool growth for new nodes) and **compress** (low-rank approximation to mitigate parameter inflation).

## Key Arguments

1. **Catastrophic forgetting is avoided by freezing the backbone.** Unlike prior CSTF methods (TrafficStream, STKEC, PECPM, TFMoE) that tune all STGNN parameters each period, EAC freezes the base model after initial training and only tunes the prompt pool thereafter.

2. **Node-level prompts capture heterogeneity.** Theoretical analysis (Proposition 1) proves that injecting node prompt parameters strictly increases the average node deviation D(X), expanding the feature space's ability to express node heterogeneity.

3. **Prompt pool exhibits low-rank property.** Spectral analysis shows >75% cumulative singular value concentration in the first few components. Proposition 2 proves that P can be approximated as AB with k = O(log(min(n,d))) when n grows large, enabling ~41% parameter reduction (k=6).

4. **Universality across STGNN architectures.** EAC improves performance consistently across spectral/spatial graph convolutions and recurrent/convolution/attention sequence operators, with recurrent-based methods benefiting most.

## Strengths

- **Simplicity**: Only a prompt pool is tuned; backbone stays frozen. Training speed accelerates 1.26–3.02× vs baselines on Energy-Stream.
- **Effectiveness**: Consistent SOTA across three real-world datasets from different domains (traffic PEMS-Stream, air quality Air-Stream, wind energy Energy-Stream) at 3/6/12-step horizons.
- **Few-shot robustness**: With only 20% training data per period, EAC shows the mildest performance decline vs all baselines.
- **Theoretical grounding**: Both expand and compress principles are formally analyzed, not just empirically observed.

## Limitations

- Parameter inflation is reduced but not eliminated; the number of prompt parameters still grows with node count. The paper acknowledges this and suggests sparsification/pruning as future work.
- All benchmarks assume graph expansion only (no node removal), though the node-level design is claimed to handle contraction flexibly.
- Maximum span is 7 years; longer periods may require full retraining.

## Relevance to Wiki

EAC is the strongest prompt-based CSTF baseline before [[stbp|STBP]] (ICLR 2026). It bridges continual learning and prompt learning in the spatio-temporal domain, establishing tuning principles later refined by STBP's pure-expansion approach. The paper was authored by the same group (HKUST-GZ, Yuxuan Liang) that later produced [[urbanfm|UrbanFM]] and [[factost|FactoST]].
