---
title: "Source: TESTAM — A Time-Enhanced Spatio-Temporal Attention Model with Mixture of Experts"
type: source-summary
tags:
  - time-series
  - spatial-temporal
  - traffic-forecasting
  - mixture-of-experts
  - attention
created: 2026-06-08
last_updated: 2026-06-08
source_count: 0
confidence: high
status: active
---

# Source: TESTAM

**Full title:** TESTAM: A Time-Enhanced Spatio-Temporal Attention Model with Mixture of Experts  
**Authors:** Hyunwook Lee, Seungmin Jin, Hyeshin Chu, Hongkyu Lim, Sungahn Ko (UNIST)  
**Venue:** ICLR 2024 (Poster, accepted)  
**Links:** [arXiv:2403.02600](https://arxiv.org/abs/2403.02600) | [GitHub](https://github.com/HyunWookL/TESTAM)  
**DOI:** Published as a conference paper at ICLR 2024

## Summary

TESTAM proposes a MoE-based spatio-temporal attention model for traffic forecasting with three specialized experts, each using a different spatial modeling method: an identity matrix (no spatial modeling, focusing on temporal patterns), a learnable adjacency matrix (static graph), and spatial attention (dynamic graph). The model routes input adaptively to the most appropriate expert via memory-augmented gating networks with two classification losses: worst-route avoidance and best-route selection.

Key method: architects combine temporal information embedding (Time2Vec) with a novel time-enhanced attention layer that transfers the model's attention domain from historical $T'$ time steps to future $T$ time steps, eliminating the need for autoregressive decoding and its associated error propagation. The memory-augmented gating networks learn to map input-output relationships directly, while the two routing classification losses (cross-entropy with pseudo labels generated from regression error quantiles) enable fine-grained routing that conventional MoE models fail to achieve in regression settings.

TESTAM achieves SOTA on all three benchmark datasets (METR-LA, PEMS-BAY, EXPY-TKY), with particularly strong performance in long-term forecasting and on the large-scale EXPY-TKY (1,843 nodes). Despite using three separate expert networks, the model has only 224K parameters — the smallest among all compared models. Ablation studies confirm that in-situ spatial modeling with diverse graph structures and both routing losses are essential to performance.

## Key Contributions

1. First MoE model for spatio-temporal traffic forecasting with heterogeneous expert architectures using three distinct spatial modeling methods
2. Time-enhanced attention mechanism that eliminates autoregressive error propagation
3. Memory-based gating networks with regression-aware routing classification losses that solve the MoE routing freeze problem in regression tasks
4. State-of-the-art performance on three real-world benchmarks, notably excelling on large-scale graphs and non-recurring traffic conditions

## Limitations

Expert specialization and routing decisions are not manually interpretable. The model has only been evaluated on three road-traffic datasets; generalization to other spatio-temporal domains (weather, pedestrian flow) is planned but unverified.
