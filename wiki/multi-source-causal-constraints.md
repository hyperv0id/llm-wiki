---
title: "Multi-Source Causal Constraints"
type: technique
tags:
  - causal-inference
  - graph-learning
  - multi-modal
  - causal-graph
created: 2026-07-14
last_updated: 2026-07-14
source_count: 1
confidence: medium
status: active
---

# Multi-Source Causal Constraints

**Multi-source causal constraints** is a causal graph learning strategy introduced by [[causalx|CausalX]] where multiple complementary causal analysis techniques jointly supervise a learned dynamic graph, covering predictive, interventional, temporal, and generative aspects of causality[^src-causalx].

## Motivation

A single causal signal is often insufficient to characterize complex multi-modal spatio-temporal dependencies, since different signals capture different aspects of the underlying relational structure[^src-causalx]. Moreover, ground-truth causal graphs are typically unavailable in real-world forecasting settings, making it impossible to rely on explicit causal labels[^src-causalx]. Multi-source constraints address this by aggregating weak but complementary signals.

## The Four Constraints

In CausalX, each constraint produces a supervision graph that is used to regularize the learned attention weights α via MSE loss[^src-causalx]:

| Constraint | Causal Aspect | Method | Edge Score |
|-----------|--------------|--------|------------|
| **Granger causality** | Predictive | F-test on lagged VAR; avg over batches of min p-value over lags | $\text{Granger}(e \to \text{tar}) = 1 - \frac{1}{B}\sum_{b=1}^{B} \min_{\ell \in [1,L]} \text{p-value}_{b,\ell}$ |
| **do-calculus** | Interventional | Mean-scale perturb node i; rerun GAT; measure embedding shift | $\text{CG}_\text{do}(i \to j) = \frac{1}{d}\sum_k \|Z_{j,k} - Z_{j,k}^{\text{per}(i)}\|$ |
| **TDMI** | Temporal | Max MI between env(e) and target(tar) over τ ∈ [1, τmax] | $\text{TDMI}(e \to \text{tar}) = \max_\tau \text{MI}(x_{\text{env}}^{t-\tau}(e), x_{\text{target}}^t(\text{tar}))$ |
| **VAE** | Generative | Encode i → latent; reconstruct j; score = −MSE(F̂ⱼ, Fⱼ) | $\text{Score}(i \to j) = -\text{MSE}(f_\text{dec}(z_i), F_j)$ |

## Complementarity Evidence

Ablation studies show that removing any single constraint degrades performance across all backbones tested[^src-causalx]. On TCNM, removing Granger causes the largest drop; on SingularTrajectory, removing do-calculus causes the largest drop — indicating task-dependent constraint importance[^src-causalx]. Per-constraint chord visualizations confirm distinct and complementary patterns: Granger and do-calculus focus on short-lag drivers, TDMI surfaces mid-lag temporal dependencies, and VAE captures diverse cross-modal couplings[^src-causalx].

## Relationship to Other Causal Approaches

Unlike SCM-based methods (e.g., [[source-cast|CaST]]) that rely on back-door/front-door adjustment over a pre-specified causal DAG, multi-source constraints treat the graph itself as the object of learning, using diverse causal analysis outputs as supervision without requiring a pre-specified causal structure[^src-causalx]. This makes the approach more modular and architecture-agnostic but places greater burden on constraint quality and coverage.

## Links

- [[causalx]] — the model that introduces this technique
- [[granger-causality]] — predictive causality constraint
- [[do-calculus]] — interventional causality constraint
- [[diffusion-model]] — used in the subsequent graph refinement stage

[^src-causalx]: [[source-causalx]]
