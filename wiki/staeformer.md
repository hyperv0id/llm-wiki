---
title: "STAEFormer"
type: entity
tags:
  - time-series
  - traffic-forecasting
  - transformer
created: 2026-05-04
last_updated: 2026-08-29
source_count: 3
confidence: low
status: active
---

# STAEFormer

**STAEFormer** was a previous state-of-the-art Transformer model for traffic forecasting, serving as a baseline for models such as [[conformer|ConFormer]] (KDD 2026). It serves as the backbone Urban Time Series Model (UTSM) in [[pn-train|PN-Train]] (ICLR 2025), which discovered and fine-tunes pattern neurons within its transformer layers to improve low-frequency pattern forecasting[^src-pn-train]. It was also one of eight backbones in [[lets-group|Let's Group]] (IJCAI 2025): attaching the SGL subgraph partition module to STAEformer's spatial feature extraction is reported to keep accuracy comparable (PEMS07 MAE 19.22→19.16, PEMS04 MAE 18.25→18.29, Table 3 of that paper) while cutting GPU cost from 22.11 to 9.34 GB on PEMS07[^src-lets-group]. Additional details pending future ingestion of the original paper.[^src-staeformer]

## Related Pages
- [[conformer]]
- [[pn-train|PN-Train]] — uses STAEformer as backbone UTSM
- [[pattern-neuron]] — the neurons discovered within STAEformer
- [[traffic-forecasting]]
- [[lets-group|Let's Group]] — uses STAEformer as one of its memory-efficiency backbones
[^src-staeformer]: [[source-staeformer]]
[^src-pn-train]: [[source-pn-train]]
[^src-lets-group]: [[source-lets-group]]