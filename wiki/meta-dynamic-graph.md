---
title: "Meta Dynamic Graph"
type: concept
tags:
  - traffic-forecasting
  - spatial-temporal
  - dynamic-graph
  - meta-learning
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Meta Dynamic Graph

The **Meta Dynamic Graph** is a design principle introduced in [[metadg|MetaDG]] (AAAI 2026) that extends the usage of dynamics in spatio-temporal models beyond spatial topology to encompass **meta-parameters** and other model intermediates[^src-metadg].

## Motivation

Traditional dynamic spatio-temporal methods like DGCRN and STSGCN model dynamics only through changing adjacency matrices — the spatial topology evolves, but the model parameters remain static[^src-metadg]. This limits the influence of dynamics to a single component of the model.

The Meta Dynamic Graph approach argues that dynamics is a more **intrinsic property** of spatio-temporal systems[^src-metadg]: if traffic conditions change, not only should the graph structure change, but also the model's internal computations (weights, parameters) should adapt to the current state. This extends dynamics to a broader scope:

| Traditional Dynamics | Meta Dynamic Graph |
|---------------------|-------------------|
| Dynamic $A^t$ only | Dynamic $A^t$ + $\theta^t$ (meta-parameters) + $\phi^t$ (edge weights) |
| Spatial only | Spatio-temporal unified |
| Graph topology focus | Full model adaptation |

## Mechanism

In [[metadg|MetaDG]], the dynamic graph structure is modeled at the level of **node representations**[^src-metadg]:

1. **Dynamic node embeddings** $N^t$ are generated at each time step from static embeddings, temporal signals, and hidden states.
2. These embeddings produce both the adjacency matrix (spatial structure) and meta-parameters (model weights) — making dynamics govern the entire forward pass, not just message-passing topology.
3. An **edge-weight adjustment matrix** $\phi^t$ further refines the graph based on message-passing reliability, adding a qualification dimension to the dynamic graph.

## Significance

The Meta Dynamic Graph concept represents a shift from "dynamics as an add-on" (generating different adjacency matrices) to "dynamics as the organizing principle" of the model[^src-metadg]. It is a concrete implementation of the [[st-unification|ST-unification]] agenda — bridging spatial and temporal dimensions by making dynamics the central mechanism through which they interact.

## Related Pages

- [[metadg]] — MetaDG model
- [[st-unification]] — ST-isolated vs ST-unification framing
- [[dynamic-graph-qualification]] — DGQ module for edge qualification
- [[gwnet]] — GWNet, static learned adjacency (no dynamics)
- [[traffic-forecasting]] — Traffic forecasting overview

[^src-metadg]: [[source-metadg]]
