---
title: "do-calculus"
type: technique
tags:
  - causal-inference
  - interventional
  - pearl
  - structural-causal-model
created: 2026-07-14
last_updated: 2026-07-17
source_count: 2
confidence: medium
status: active
---

# do-calculus

**do-calculus** (Pearl, 1994) is a formal framework for reasoning about interventions in causal graphical models. The do-operator $\text{do}(X := x)$ represents an external intervention that sets variable X to value x, breaking its incoming causal edges — distinct from conditioning (seeing) on X = x[^src-causalx].

## Core Idea

In a structural causal model (SCM), $\text{do}(X := x)$ modifies the data-generating process itself: the structural equation for X is replaced by the constant x, while all other mechanisms remain unchanged. do-calculus provides three rules for transforming expressions involving the do-operator into standard probabilistic expressions, enabling interventional and counterfactual queries from observational data, provided the causal DAG is known[^src-causalx].

## Operationalization in Deep Learning (CausalX)

In [[causalx|CausalX]], do-calculus is operationalized as a feature-level intervention score for causal graph learning[^src-causalx]:

1. For each node i, **intervene** $\text{do}(F_i)$ by replacing its feature with the sample-wise mean across all nodes: $\tilde{F}_{:,i,:} = \frac{1}{TN}\sum_m F_{:,m,:}$
2. Keep all other node features unchanged: $\tilde{F}_{:,m,:} = F_{:,m,:}$ for $m \neq i$
3. Re-run the same GAT on the perturbed graph to obtain new embeddings $Z^{\text{per}(i)}$
4. The interventional effect of node i on node j is: $\text{CG}_\text{do}(i \to j) = \frac{1}{d}\sum_{k=1}^d |Z_{j,k} - Z_{j,k}^{\text{per}(i)}|$

This produces a batch-level causal matrix used to supervise the learned attention α via MSE loss, complementing Granger's predictive view with an interventional (counterfactual) perspective[^src-causalx].

## Role in Multi-Source Causal Constraints

In CausalX's [[multi-source-causal-constraints|multi-source causal constraints]] framework, do-calculus provides the **interventional** constraint — it captures how perturbing one node causally affects others through the learned GNN, rather than mere predictive association. Ablation shows it is particularly influential for pedestrian trajectory (SingularTrajectory: largest degradation when removed) compared to Granger's dominance on TC forecasting, suggesting task-dependent importance of interventional vs. predictive signals[^src-causalx].

## Distinction from Related Concepts

| Concept | Question |
|---------|----------|
| **Granger causality** | Does X's past help predict Y's future? (predictive) |
| **do-calculus / interventional** | What happens to Y if I force X to some value? (interventional) |
| **Counterfactual** | What would Y have been, given what actually happened? |

do-calculus requires stronger assumptions than Granger (knowledge of the causal graph structure), but answers fundamentally different questions. CausalX operationalizes do-calculus without requiring a pre-specified causal DAG by using the learned GNN itself as a proxy for the structural equations[^src-causalx].

## Links

- [[causalx]] — operationalizes do-calculus for causal graph learning
- [[granger-causality]] — predictive complement
- [[multi-source-causal-constraints]] — framework combining do-calculus with other constraints
- [[causal-time-series-forecasting]] — broader causal TS paradigm including interventional queries
- [[causal-counterfactual-recovery]] — DoFlow's CNF-based counterfactual recovery (also uses Pearl's framework)
- [[e2-cstp|E²-CSTP]] — applies backdoor adjustment (derived from do-calculus) for multi-modal ST causal inference (NeurIPS 2025)
- [[backdoor-adjustment]] — the specific do-calculus application for blocking confounding paths [^src-e2-cstp]

[^src-causalx]: [[source-causalx]]
[^src-e2-cstp]: [[source-e2-cstp]]
