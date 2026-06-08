---
title: "Memory-Augmented Gating"
type: technique
tags:
  - mixture-of-experts
  - routing
  - memory-network
  - spatial-temporal
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Memory-Augmented Gating

Memory-augmented gating is a routing mechanism for Mixture of Experts (MoE) models, introduced in [[testam|TESTAM]], that uses a learnable memory bank to compute expert routing probabilities from input-output similarity, combined with two classification losses to enable fine-grained routing in regression settings[^src-testam].

## Problem

Conventional MoE models suffer from a critical limitation in regression tasks: the gating network barely changes its routing decisions after initialization because the gate is not properly guided by regression loss gradients[^src-testam]. This causes "mismatches" — uninformative and unchanging routing — preventing experts from specializing.

## Mechanism

TESTAM's gating works in two stages[^src-testam]:

### 1. Memory Querying

For each node $i$ at time step $t$, the input $X_i^{(t)}$ is projected through a linear layer and matched against a meta-node bank $M \in \mathbb{R}^{m \times e}$ (with $m$ memory items of dimension $e$) via softmax similarity:

$$a_j = \frac{\exp(Q_i^{(t)} M[j]^\top)}{\sum_{j=1}^{m} \exp(Q_i^{(t)} M[j]^\top)}, \quad O_i^{(t)} = \sum_{j=1}^{m} a_j M[j]$$

### 2. Routing Classification

Given expert output $z_e$ and queried memory $O_i^{(t)}$, the routing probability is computed as $p_e = \text{softmax}(g(z_e, O_i^{(t)}))$, where $g$ is a similarity function. Two cross-entropy losses guide training[^src-testam]:

- **Worst-route avoidance loss** ($\mathcal{L}_{\text{worst}}$): Point-wise loss that penalizes routing when the selected expert's prediction error exceeds a $q$-th quantile threshold. Pseudo labels assign zero to incorrectly selected experts and $1/(E-1)$ to unselected experts.
- **Best-route selection loss**: Node-wise loss that rewards routing when the selected expert achieves error below the $(1-q)$-th quantile. More challenging to train than worst-route avoidance due to traffic data noise.

Both losses use $q=0.7$, and ablation shows that excluding the best-route selection loss degrades performance (METR-LA MAE 2.96 vs 2.93)[^src-testam].

## Significance

This technique solves the MoE routing freeze problem in regression by transforming routing into a supervised classification task with learnable pseudo labels. The memory bank learns typical input-output relationships, enabling context-aware routing. Ablation with "w/o gating" (no expert selection) degrades to GMAN-level performance, confirming that adaptive routing — not model capacity — drives TESTAM's improvement[^src-testam].

[^src-testam]: [[source-testam]]
