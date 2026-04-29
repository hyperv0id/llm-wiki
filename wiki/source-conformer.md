---
title: "ConFormer: Conditional Transformer for Accident-Informed Traffic Forecasting"
type: source-summary
tags:
  - traffic-forecasting
  - transformer
  - accident-aware
  - kdd-2026
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# ConFormer: Conditional Transformer for Accident-Informed Traffic Forecasting

**Authors**: Hongjun Wang, Jiawei Yong, Jiawei Wang, Shintaro Fukushima, Renhe Jiang (The University of Tokyo, Toyota Motor Corporation)

**Venue**: KDD 2026, Jeju Island, Republic of Korea

## Core Contribution

ConFormer addresses a critical gap in traffic forecasting: existing models excel at capturing recurring patterns but fail when traffic accidents create non-stationary perturbations with directional shockwaves through transportation networks[^src-conformer].

## Key Innovations

1. **Accident-Aware Graph Propagation** — Models how disruptions spread asymmetrically through traffic networks using graph convolution with K-hop Laplacian operations[^src-conformer].

2. **Guided Layer Normalization (GLN)** — Replaces static LayerNorm parameters with dynamic affine transformations (γ, β) conditioned on traffic conditions, enabling adaptive feature transformations across normal and accident scenarios[^src-conformer].

3. **Conditional Self-Attention** — Extends vanilla self-attention to incorporate contextual condition representations, with residual connections modulated by learned factor α[^src-conformer].

## Datasets

Two enriched large-scale benchmark datasets:
- **Tokyo**: 1,843 highway segments, Oct-Dec 2021, 10-min intervals, accident + regulation data from JARTIC
- **California**: Bay Area (2,352 sensors) + San Diego (716 sensors), 2019, 15-min intervals, accident data from US Accidents database[^src-conformer]

## Performance

ConFormer consistently outperforms state-of-the-art models including STAEFormer:
- Tokyo: 1.7% MAE improvement, 21.5% MAPE improvement in accident scenarios
- San Diego: 4.7% MAE improvement, 5.0% MAPE improvement
- Bay Area: 1.8% MAE improvement
- Up to **10.7% improvement** in accident scenarios specifically[^src-conformer]

## Theoretical Insight

GLN enables adaptive feature transformations through condition-dependent affine parameters. The scaling factor γ controls sensitivity to abrupt changes (higher γ → rapid accident adaptation), while the shifting factor β emphasizes node-specific features during disruptions[^src-conformer].

## Limitations

- Requires accident/regulation data integration
- Single attention layer may limit very long-range dependencies
- Graph propagation order (K) adds computational overhead

[^src-conformer]: [[source-conformer]]