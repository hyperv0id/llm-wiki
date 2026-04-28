---
title: "HyperD: Hybrid Periodicity Decoupling Framework for Traffic Forecasting"
type: source-summary
tags:
  - traffic-forecasting
  - periodicity-decoupling
  - spatial-temporal-gnn
  - fourier-transform
  - time-series
created: 2026-04-27
last_updated: 2026-04-27
source_count: 1
confidence: high
status: active
---

# HyperD: Hybrid Periodicity Decoupling Framework for Traffic Forecasting

Hao Wen, Nan Feng (Jilin University), 2025. arXiv 2511.09275.

## Core Claim

Traffic forecasting models should explicitly **decouple short-term and long-term periodicity** rather than modeling them implicitly or uniformly. By decomposing traffic signals in the frequency domain and using specialized encoders for each periodic scale, forecasting accuracy improves substantially, especially on long horizons[^src-hyperd-hybrid-periodicity-decoupling].

## Key Contributions

1. **Hybrid Periodicity Decoupling framework (HyperD)** — the first framework to explicitly separate short-term (intra-day, recent) and long-term (daily/weekly) periodicity with dedicated architectural components[^src-hyperd-hybrid-periodicity-decoupling].

2. **Frequency-Aware Residual Representation (FR)** — decomposes traffic signals via Real Fast Fourier Transform (RFFT), clustering frequency bands into residual, short-term, and long-term components with learned thresholds[^src-hyperd-hybrid-periodicity-decoupling].

3. **Spatial-Temporal Attentive Encoder (STAE)** — dual-pathway design where the short-term encoder processes the original signal while the long-term encoder processes the residual signal (with short-term fluctuations removed via FR)[^src-hyperd-hybrid-periodicity-decoupling].

4. **Two-stage Decoupled MLP Decoder (fDMLP)** — coarse-to-fine prediction with explicit trend removal to prevent over-smoothing[^src-hyperd-hybrid-periodicity-decoupling].

5. **Dual-View Alignment (DVA) Loss** — aligns short-term and long-term representations across time scales (via Soft-DTW) and periodicity views (via cosine similarity) to ensure consistent yet complementary representations[^src-hyperd-hybrid-periodicity-decoupling].

## Results

HyperD achieves state-of-the-art MAE, RMSE, and MAPE on all four PeMS datasets (PEMS03, 04, 07, 08), outperforming 15 baselines including STAEformer, D2STGNN, FreTS, and PDFormer[^src-hyperd-hybrid-periodicity-decoupling]. The performance gap widens at longer horizons (9-step, 12-step), validating the benefit of explicit periodicity decoupling for long-range forecasting. Ablation studies confirm each component (FR, dual-pathway, DVA loss, fDMLP) contributes meaningfully, with the FR module being the most critical.

## Limitations

- Learned frequency cutoffs are fixed per dataset, not adaptive to individual nodes or time windows[^src-hyperd-hybrid-periodicity-decoupling].
- Does not fully integrate graph convolutions (uses spatial embeddings only)[^src-hyperd-hybrid-periodicity-decoupling].
- No incorporation of external factors (weather, holidays, accidents)[^src-hyperd-hybrid-periodicity-decoupling].

[^src-hyperd-hybrid-periodicity-decoupling]: [[source-hyperd-hybrid-periodicity-decoupling]]
