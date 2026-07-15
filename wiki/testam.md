---
title: "TESTAM"
type: entity
tags:
  - time-series
  - spatial-temporal
  - traffic-forecasting
  - mixture-of-experts
  - attention
  - iclr-2024
created: 2026-06-08
last_updated: 2026-07-16
source_count: 2
confidence: high
status: active
---

# TESTAM

TESTAM (Time-Enhanced Spatio-Temporal Attention Model) is a MoE-based spatio-temporal attention framework for traffic forecasting, published at ICLR 2024[^src-testam]. It is the first model to combine Mixture of Experts with heterogeneous expert architectures for in-situ spatial modeling in traffic forecasting.

## Architecture

TESTAM uses three parallel transformer-style experts, each with an identical encoder structure (temporal attention, spatial modeling, time-enhanced attention, point-wise FFN) but differing only in their spatial modeling layer[^src-testam]:

| Expert | Spatial Method | Role |
|--------|---------------|------|
| Identity | No spatial mixing ($I$) | Captures temporal-only and self-referencing patterns |
| Static Graph | Learnable adjacency matrix ($EWE^\top$) | Models stable, recurring spatial dependencies |
| Dynamic Graph | Full spatial attention | Captures time-varying, similarity-based spatial relationships |

Each expert receives Time2Vec temporal information embedding concatenated with input features. The [[time-enhanced-attention]] layer replaces standard decoder cross-attention, eliminating autoregressive error propagation.

### Gating Networks

Routing is handled by [[memory-augmented-gating]] networks that query a meta-node bank $M \in \mathbb{R}^{m \times e}$ to compute routing probabilities from input-output similarity, trained with two cross-entropy classification losses using regression error-based pseudo labels[^src-testam].

## Performance

TESTAM achieves SOTA across all three benchmarks[^src-testam]:

| Dataset | Nodes | Interval | MAE | Improvement Over Best Baseline |
|---------|-------|----------|-----|-------------------------------|
| METR-LA | 207 | 5 min | 2.93 | ~1-3% |
| PEMS-BAY | 325 | 5 min | 1.53 | ~2-4% |
| EXPY-TKY | 1,843 | 10 min | 6.40 | ~3-6% |

The model is particularly strong on large-scale graphs (EXPY-TKY, 1,843 nodes) and non-recurring traffic conditions (holidays, accidents, traffic controls)[^src-testam]. Despite using three experts, TESTAM has only **224K parameters** — the fewest among compared models — and achieves the second-fastest inference (7.96s)[^src-testam].

## Key Innovations

1. **In-situ spatial modeling**: Different traffic conditions benefit from different spatial priors; TESTAM routes adaptively rather than committing to one[^src-testam].
2. **Time-enhanced attention**: Eliminates autoregressive decoding by transferring attention from source to target time steps[^src-testam].
3. **Regression-aware MoE routing**: Two classification losses prevent the routing freeze problem known in regression MoE settings[^src-testam].

## Related

- [[pn-train|PN-Train]] — neuron-level training method that compares against TESTAM (ICLR 2025)
- [[pattern-neuron]] — pattern neurons discovered in UTSMs
- [[traffic-forecasting]] — the task domain
- [[time-enhanced-attention]] — the core attention mechanism
- [[memory-augmented-gating]] — the routing mechanism
- [[mixture-of-experts-routing]] — broader MoE routing paradigm
- See also: [[gwnet|GWNet]], [[dcrnn|DCRNN]], [[source-astgcn|ASTGCN]], [[hephestus|HEPHAESTUS]]

[[stamimputer|STAMImputer]] (arXiv 2025) 是 MoE-based 时空注意力的姊妹工作，但面向**填补**而非预测：专家按维度分工（时间 vs 空间），路由由观测专家依据稀疏度特征裁决，区别于 TESTAM 的记忆增强门控与异质 expert 设计[^src-stamimputer]。

[^src-testam]: [[source-testam]]
[^src-stamimputer]: [[source-stamimputer]]
