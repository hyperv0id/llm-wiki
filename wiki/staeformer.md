---
title: "STAEFormer"
type: entity
tags:
  - time-series
  - traffic-forecasting
  - transformer
created: 2026-05-04
last_updated: 2026-07-16
source_count: 2
confidence: low
status: active
---

# STAEFormer

**STAEFormer** was a previous state-of-the-art Transformer model for traffic forecasting, serving as a baseline for models such as [[conformer|ConFormer]] (KDD 2026). It serves as the backbone Urban Time Series Model (UTSM) in [[pn-train|PN-Train]] (ICLR 2025), which discovered and fine-tunes pattern neurons within its transformer layers to improve low-frequency pattern forecasting[^src-pn-train]. Additional details pending future ingestion of the original paper.[^src-staeformer]

## Related Pages
- [[conformer]]
- [[pn-train|PN-Train]] — uses STAEformer as backbone UTSM
- [[pattern-neuron]] — the neurons discovered within STAEformer
- [[traffic-forecasting]]
[^src-staeformer]: [[source-staeformer]]
[^src-pn-train]: [[source-pn-train]]