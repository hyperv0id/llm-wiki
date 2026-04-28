---
title: "ConFormer"
type: entity
tags:
  - traffic-forecasting
  - transformer
  - conditional-modeling
  - kdd-2026
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# ConFormer

**ConFormer** (Conditional Transformer) is a traffic forecasting model that explicitly models the disruptive impact of accidents on transportation networks. Proposed in KDD 2026, it addresses the critical limitation that existing approaches excel at capturing recurring patterns but falter when confronted with non-stationary perturbations induced by traffic accidents[^src-conformer].

## Architecture

ConFormer integrates two key innovations:

1. **Accident-Aware Graph Propagation** — Uses K-hop graph Laplacian operations to model how disruptions spread asymmetrically through traffic networks[^src-conformer].

2. **[[guided-layer-normalization|Guided Layer Normalization (GLN)]]** — Replaces static LayerNorm parameters with dynamic affine transformations conditioned on traffic conditions[^src-conformer].

3. **Conditional Self-Attention** — Extends vanilla self-attention to incorporate contextual condition representations[^src-conformer].

## Related Concepts

- [[accident-aware-traffic-forecasting]] — the problem domain ConFormer addresses
- [[staeformer|STAEFormer]] — previous SOTA transformer for traffic forecasting
- [[hyperd|HyperD]] — periodicity-decoupled traffic forecasting
- [[timesnet|TimesNet]] — general time series foundation model

[^src-conformer]: [[source-conformer]]