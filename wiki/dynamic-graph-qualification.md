---
title: "Dynamic Graph Qualification"
type: technique
tags:
  - traffic-forecasting
  - graph-neural-network
  - dynamic-graph
  - message-passing
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Dynamic Graph Qualification (DGQ)

**Dynamic Graph Qualification (DGQ)** is a module in [[metadg|MetaDG]] (AAAI 2026) that refines the dynamically generated adjacency matrix by qualifying the reliability of information propagation on edges[^src-metadg].

## Motivation

In GCRU-based models, information propagated on the graph includes both current-step input and previous-step hidden state. Since the recurrent nature of GRU may lead to error accumulation, qualifying graph convolution reliability is even more important than in standard GCNs[^src-metadg].

The key insight: **cross-time-step similarity** between node representations indicates reliable information channels — if two nodes consistently interact across time steps, their edges should be strengthened[^src-metadg].

## Mechanism

### Step 1: Edge Qualification Matrix $P^t$

$$P^t = \text{asym}(\text{ReLU}(M \odot (N^m_t \cdot N^{m\top}_{t-1})))$$

- $N^m_t, N^m_{t-1}$: enhanced node representations at consecutive time steps
- $M$: static 0-1 adjacency mask (via learnable node embedding inner product), limiting dynamics to physically connected edges
- $\text{asym}(\cdot)$: row normalization[^src-metadg]

### Step 2: Node-Wise Threshold $\epsilon_t$

$$\epsilon_{t; i} = P_{t; (i,i)} \sigma(N^m_{t; i} \cdot \epsilon), \quad \forall v_i \in V$$

The threshold baseline is $P_{t; (i,i)}$ (self-loop qualification), ensuring self-loops are never weakened — stabilizing training[^src-metadg].

### Step 3: Proportional Strengthen, Fixed Weaken

Edges above threshold → strengthened proportionally; edges below → weakened[^src-metadg]:

$$\phi^t = \beta^t \odot M^{pos}_t + \beta^t \odot M^{neg}_t$$

where $M^{pos}_t$ and $M^{neg}_t$ are threshold-comparison masks, and $\beta^t = \exp(\text{InstanceNorm}(M^{pos}_t) \cdot \delta)$ are adaptive scaling coefficients[^src-metadg]. The strategy follows UnGSL's "proportional strengthen, fixed weaken" approach but with **adaptive** coefficients rather than fixed scalars, accounting for dynamic graph complexity[^src-metadg].

### Final Qualified Graph

$$\tilde{A}^t = \text{asym}(\phi^t \odot A^t)$$

The raw adjacency matrix $A^t$ is element-wise multiplied by $\phi^t$, then row-normalized for use in graph convolution[^src-metadg].

## Ablation Evidence

Removing DGQ consistently degrades performance across all four PEMS datasets[^src-metadg]:
- PEMS03: MAE 14.48 vs 14.29 (full MetaDG, ↑1.3%)
- PEMS07: MAE 18.91 vs 18.79 (↑0.6%)
- PEMS08: MAE 13.06 vs 13.04 (↑0.2%)

The relatively small magnitude suggests DGQ provides fine-grained refinement rather than coarse structural change — consistent with its role as a qualification/calibration module[^src-metadg].

## Related Pages

- [[metadg]] — MetaDG model
- [[meta-dynamic-graph]] — The broader concept
- [[st-unification]] — ST-unification framing
- [[traffic-forecasting]] — Traffic forecasting overview

[^src-metadg]: [[source-metadg]]
