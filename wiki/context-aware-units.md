---
title: "Context-Aware Units (ConAU)"
type: technique
tags:
  - spatio-temporal
  - attention
  - out-of-distribution
  - representation-learning
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Context-Aware Units (ConAU)

**Context-Aware Units (ConAU)** are the K learnable feature vectors that serve as the shared interaction hub in [[stop|STOP]]'s [[centralized-message-passing|centralized message passing]] mechanism (ICML 2025)[^src-stop]. Each ConAU is a learnable vector c ∈ R^{d}, collected into a bank C = [c₁, …, c_K] ∈ R^{K×d}, where K ≪ N (the number of graph nodes) is a hyperparameter[^src-stop].

## Function

ConAUs are designed to **perceive generalizable contextual features** from nodes[^src-stop]. Through [[centralized-message-passing|multi-head low-rank attention]], nodes' features are first *aggregated* into the K context vectors, which are then *diffused* back to nodes for feature interaction[^src-stop]. Because every node reads from the same small, stable set of context units rather than from its (possibly changed) neighbors, the learned representation is resilient to structural shift and transfers to newly added nodes — the basis of STOP's inductive-learning capability[^src-stop].

After interaction, STOP isolates a node's **personalized features** by subtracting the shared contextual features from its temporal representation (Zp = Z_T − Z_c), then recombines personalized and contextual features; visualizations show contextual features capture shared node patterns while personalized features tailor per-node predictions[^src-stop].

## Hyperparameter sensitivity

K is critical. Too many ConAUs (large K) prevents the model from focusing on invariant contextual features and injects noise; too few cannot capture sufficient invariant knowledge[^src-stop]. The paper uses K ∈ {8, 24, 32, 64, 8, 4} across its six datasets[^src-stop]. Ablating ConAU entirely ("w/o ConAU") sharply increases error, confirming spatial interaction remains necessary even in OOD scenarios[^src-stop].

[^src-stop]: [[source-stop]]
