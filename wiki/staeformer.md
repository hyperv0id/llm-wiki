---
title: "STAEFormer"
type: entity
tags:
  - time-series
  - traffic-forecasting
  - transformer
created: 2026-05-04
last_updated: 2026-08-30
source_count: 4
confidence: low
status: active
---

# STAEFormer

**STAEFormer** was a previous state-of-the-art Transformer model for traffic forecasting, serving as a baseline for models such as [[conformer|ConFormer]] (KDD 2026). It serves as the backbone Urban Time Series Model (UTSM) in [[pn-train|PN-Train]] (ICLR 2025), which discovered and fine-tunes pattern neurons within its transformer layers to improve low-frequency pattern forecasting[^src-pn-train]. It was also one of eight backbones in [[lets-group|Let's Group]] (IJCAI 2025): attaching the SGL subgraph partition module to STAEformer's spatial feature extraction is reported to keep accuracy comparable (PEMS07 MAE 19.22→19.16, PEMS04 MAE 18.25→18.29, Table 3 of that paper) while cutting GPU cost from 22.11 to 9.34 GB on PEMS07[^src-lets-group]. Additional details pending future ingestion of the original paper.[^src-staeformer]

Follow-up work [[stgformer|STGformer]] (arXiv:2410.00385v2, 2024) reuses STAEformer's data embedding layer — including the spatio-temporal positional encoding Xste — and reports a 100× speedup with 99.8% GPU-memory reduction versus STAEformer in batch inference on the 8,600-sensor California graph, alongside better reported accuracy on LargeST (e.g., San Diego average MAE 17.36 vs 18.01), PEMS03/04/07/08, and cross-year (2019→2020) tests[^src-stgformer]. The STGformer paper characterizes STAEformer as relying on stacked 2L layers of spatiotemporally separable attention to obtain higher-order interactions[^src-stgformer].

## Related Pages
- [[conformer]]
- [[pn-train|PN-Train]] — uses STAEformer as backbone UTSM
- [[pattern-neuron]] — the neurons discovered within STAEformer
- [[traffic-forecasting]]
- [[lets-group|Let's Group]] — uses STAEformer as one of its memory-efficiency backbones
- [[stgformer|STGformer]] — follow-up graph transformer reusing STAEformer's embedding; reports 100× speedup / 99.8% memory reduction over STAEformer at 8,600 nodes
[^src-staeformer]: [[source-staeformer]]
[^src-pn-train]: [[source-pn-train]]
[^src-lets-group]: [[source-lets-group]]
[^src-stgformer]: [[source-stgformer]]