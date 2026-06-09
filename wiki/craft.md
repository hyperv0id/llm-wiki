---
title: "CRAFT"
type: entity
tags:
  - traffic-flow-generation
  - diffusion-model
  - cross-city
  - retrieval-augmented-generation
  - neurips-2025
created: 2026-06-08
last_updated: 2026-06-09
source_count: 2
confidence: high
status: active
---

# CRAFT

**CRAFT** (Cross-city Retrieval-Augmented traffic Flow generaTion) is a DDPM-based diffusion model for **zero-shot cross-city traffic flow generation**, published at NeurIPS 2025[^src-craft]. It enables generating realistic dynamic traffic flow data for entirely unseen cities without requiring any historical flow records — using only publicly available geographic features (POIs, roads, population).

## Core Innovation

CRAFT addresses two fundamental challenges in cross-city flow generation[^src-craft]:

1. **Domain Shift**: Geographic features that predict similar flow within one city may map to very different flows across cities. Solved by [[geographic-feature-alignment|GFA]].
2. **Insufficient Condition**: Static geography alone cannot capture temporal dynamics. Solved by [[retrieval-based-condition-augmentation|RCA]].

Both modules are lightweight plug-ins requiring no modifications to the DDPM backbone[^src-craft].

## Architecture

```
Source Cities (geo + flow) + Target City (geo only)
   │
   ├── GFA: Geographic Feature Alignment
   │      ├── Basic geo-features → Graph Transformer spatial encoder
   │      ├── TFA (Traffic Flow Alignment): region rep ∝ flow similarity
   │      └── CCA (Cross-City Alignment): optimal transport between cities
   │
   ├── RCA: Retrieval-based Condition Augmentation
   │      ├── Time embedding (month/day/hour)
   │      ├── Retrieve top-K similar flow segments from source cities
   │      └── Self-attention aggregation → augmented condition
   │
   └── Conditional Diffusion Backbone (DDPM + 1D-U-Net)
          └── Condition c_i,t = MLP(h_i ∥ x_i,t ∥ t_emb)
```

## Key Results

| Metric | CRAFT vs Avg Baseline | vs 2nd Best (GMEL) | vs DDPM |
|--------|-----------------------|---------------------|---------|
| Overall | +59.7% | +22.5% | +61.5% |
| Downstream Pred (LSTM/Transformer) | +55.9% | +14.9% (vs DDPM) | - |

- **4 cities** (Chicago, DC, Toronto, NYC): leave-one-out cross-city evaluation[^src-craft]
- **Only 10.4% degradation** vs. training on real target city data (3.8%–22.2%)[^src-craft]
- **GFA is the most critical component** (domain shift is the dominant challenge)[^src-craft]
- **Temporal embedding** contributes most within RCA, highlighting periodic pattern importance[^src-craft]

## Ablation Findings

- Removing GFA (w/o Alignment): significant performance drop across all metrics — confirms domain shift as the primary bottleneck[^src-craft]
- Removing RCA (w/o RCA): notable decline, especially in NMAE/NRMSE — confirms value of retrieved dynamics[^src-craft]
- Removing temporal embedding (w/o Time Emb): largest drop within RCA — periodic patterns essential[^src-craft]
- TFA aligns geo-features with flow distributions; CCA reduces cross-city distribution shift (t-SNE verified)[^src-craft]

## Comparison

Unlike [[rast|RAST]] (AAAI 2026) which uses RAG for traffic **prediction** within the same city, CRAFT uses RAG for cross-city traffic flow **generation** — a fundamentally different task with a stronger emphasis on zero-shot transfer[^src-craft]. Unlike foundation models like [[most|MoST]] or [[unist|UniST]] which predict from historical data, CRAFT **generates** flow data from scratch for cities with no historical records[^src-craft].

Unlike [[ratd|RATD]] (NeurIPS 2024), the first retrieval-augmented time series diffusion model that retrieves k-NN references to guide denoising for univariate/multivariate forecasting, CRAFT applies retrieval augmentation to cross-city flow generation rather than same-distribution forecasting[^src-ratd].

## Limitations

- Validated on in-out flow only; OD flows not explored[^src-craft]
- Temporal horizon ≤ 168 steps[^src-craft]
- Four North American bicycle-sharing cities[^src-craft]

## See Also

- [[source-craft]] — full source summary
- [[geographic-feature-alignment]] — GFA technique
- [[retrieval-based-condition-augmentation]] — RCA technique
- [[cross-city-traffic-flow-generation]] — problem domain overview
- [[rast]] — RAST, RAG for traffic prediction (AAAI 2026)
- [[diffusion-models]] — diffusion model concept page
- [[traffic-forecasting]] — general traffic forecasting
- [[spatio-temporal-foundation-model]] — ST foundation model landscape

[^src-craft]: [[source-craft]]
[^src-ratd]: [[source-ratd]]
