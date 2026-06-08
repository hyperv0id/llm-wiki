---
title: "Cross-City Traffic Flow Generation"
type: concept
tags:
  - traffic-flow
  - cross-city
  - zero-shot
  - generative-model
  - domain-adaptation
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Cross-City Traffic Flow Generation

**Cross-city traffic flow generation** is the task of synthesizing realistic, dynamic traffic flow data for a city without using any historical flow records from that city. Models are trained exclusively on data from source cities (which have both geographic features and flow records) and deployed in zero-shot fashion to unseen target cities (which have only geographic features)[^src-craft].

## Problem Formulation

Given:
- **Source cities**: geographic features $G^{(s)}$ + traffic flow data $X^{(s)}$
- **Target city**: geographic features $G^{(t)}$ only

Goal: Train model $F$ with parameters $\theta^{(s)}$ on source cities, then generate flow for target city:

$$\hat{X}^{(t)} = F(G^{(t)}; \theta^{(s)})$$

## Why It Matters

1. **Data scarcity**: Most cities lack large-scale traffic flow datasets due to high collection costs and privacy constraints[^src-craft]
2. **Urban planning**: Generated flow data enables downstream applications (traffic prediction, infrastructure planning, emergency management) in data-scarce cities[^src-craft]
3. **Model deployment**: Eliminates the need to train new models per city — one model serves many[^src-craft]

## Key Challenges

### Domain Shift

Within a single city, regions with similar geographic features (POI distribution, road density, population) exhibit similar traffic patterns. However, this correspondence breaks in cross-city settings due to urban heterogeneity — different city layouts, transportation cultures, and economic structures cause regions with identical geo-features to have vastly different traffic behaviors[^src-craft].

### Insufficient Condition

Static geographic features (POIs, roads, population) can capture spatial patterns and coarse periodicity but cannot convey **dynamic stochastic properties** — absolute flow volumes, peak magnitudes, or variance[^src-craft]. These dynamics are typically provided by historical flow records, which are absent for the target city.

## Approaches

### CRAFT (NeurIPS 2025)

[[craft|CRAFT]] is the first model explicitly designed for this task. It uses a DDPM-based diffusion backbone with two plug-in modules[^src-craft]:

- [[geographic-feature-alignment|GFA]]: Aligns cross-city geographic representations via Traffic Flow Alignment + Cross-City Alignment (optimal transport)
- [[retrieval-based-condition-augmentation|RCA]]: Augments diffusion conditions by retrieving similar historical flow patterns from source cities

CRAFT achieves 59.7% improvement over baseline average across 4 cities (Chicago, DC, Toronto, NYC)[^src-craft].

## Relationship to Related Tasks

| Task | Input | Output | Historical data in target? |
|------|-------|--------|---------------------------|
| **Cross-city flow generation** | Geo features | Flow data | No |
| Traffic flow prediction | Geo + historical flow | Future flow | Yes |
| [[cross-city-traffic-flow-generation|Cross-city prediction]] | Geo + source flow | Target flow | Yes (source only) |
| [[spatio-temporal-foundation-model|ST foundation models]] | Multi-modal | Predictions | Often yes |

## Open Questions

- Generalization beyond in-out flows to OD matrices[^src-craft]
- Performance on cities with substantially different urban structures (e.g., Asian vs. North American layouts)
- Longer temporal horizons (beyond 168 steps)[^src-craft]
- Integration with other RAG paradigms (e.g., [[retrieval-augmented-spatio-temporal-forecasting|spatio-temporal retrieval stores]])

## See Also

- [[craft]] — CRAFT model
- [[geographic-feature-alignment]] — GFA technique
- [[retrieval-based-condition-augmentation]] — RCA technique
- [[traffic-forecasting]] — general traffic prediction
- [[spatio-temporal-foundation-model]] — ST foundation model landscape
- [[retrieval-augmented-spatio-temporal-forecasting]] — RAG-for-STF paradigm

[^src-craft]: [[source-craft]]
