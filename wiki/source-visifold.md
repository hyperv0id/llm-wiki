---
title: "Source: VisiFold — Long-Term Traffic Forecasting via Temporal Folding Graph and Node Visibility"
type: source-summary
tags:
  - traffic-forecasting
  - spatial-temporal
  - transformer
  - long-term-forecasting
  - tokenization
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Source: VisiFold

**Full Title:** VisiFold: Long-Term Traffic Forecasting via Temporal Folding Graph and Node Visibility
**Authors:** Zhiwei Zhang, Xinyi Du, Weihao Wang, Xuanchi Guo, Wenjuan Han (Beijing Jiaotong University / Beijing Normal University)
**Venue:** arXiv:2603.11816, March 2026
**Code:** https://github.com/PlanckChang/VisiFold

## Summary

VisiFold addresses the core bottlenecks of long-term traffic forecasting: escalating computational cost and increasingly complex spatial-temporal dependencies. The authors identify two fundamental flaws in the conventional spatial-temporal graph paradigm: **snapshot-stacking inflation** (GPU memory/runtime growing rapidly with time steps T) and **cross-step fragmentation** (temporal dependencies partitioned across separate snapshots, conveyed only through intermediate representations)[^src-visifold].

The key innovation is the **Temporal Folding Graph (TFG)**, which collapses all attributes of a node across a sequence of snapshots into a single enriched token, compressing T snapshots into a single graph representation. The token count is reduced from N×T to N, eliminating the need for separate spatial and temporal modules and avoiding cross-step message passing entirely. This design increases information density at the representation level and enables synchronized spatial-temporal modeling[^src-visifold].

To prevent the resulting node count N from becoming the new bottleneck in large road networks, the authors introduce **Node Visibility** — two complementary mechanisms: (1) node-level masking, which randomly hides a subset of nodes from the encoder (inspired by MAE), and (2) subgraph sampling, which partitions remaining nodes into fixed-size subgraphs to increase parallelism. Beyond efficiency gains, node visibility serves as an implicit regularizer that discourages position-dependent shortcuts and forces the model to learn more robust adjacency-insensitive representations[^src-visifold].

The overall **VisiFold** architecture concatenates TFG-derived token embeddings with spatial, time-of-day, and day-of-week embeddings, applies node visibility during training only, then processes refined representations through a standard Transformer encoder with an MLP prediction head. Huber loss is used for training[^src-visifold].

Extensive experiments on PEMS04, PEMS08, and SEATTLE datasets (24/36/48-step horizons) vs. 12 baselines demonstrate SOTA accuracy across all scenarios. Resource efficiency is remarkable: ~7× training speedup and ~4× GPU memory saving vs. STAEformer, with inference under one second — enabling real-time deployment. Notably, the model maintains performance even with 80% of nodes masked (r=0.8), revealing substantial redundancy in traffic data. Ablation confirms spatial embeddings as the dominant accuracy driver and TFG significantly outperforming the spatial folding (SF) alternative[^src-visifold].

## Limitations

The model is purely data-driven and cannot respond to emergent events (accidents, closures). The authors suggest integrating multi-modal metadata and temporal modules into the TFG framework as future directions[^src-visifold].

[^src-visifold]: [[source-visifold]]