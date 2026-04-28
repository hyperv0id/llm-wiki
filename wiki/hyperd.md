---
title: "HyperD"
type: entity
tags:
  - framework
  - traffic-forecasting
  - periodicity-decoupling
  - fourier-transform
created: 2026-04-27
last_updated: 2026-04-28
source_count: 5
confidence: high
status: active
---

# HyperD

HyperD (Hybrid Periodicity Decoupling) is a deep learning framework for traffic forecasting, proposed by Hao Wen and Nan Feng at Jilin University in 2025[^src-hyperd-hybrid-periodicity-decoupling]. It explicitly decouples short-term and long-term periodicity from traffic signals and models each with specialized architectural components.

## Architecture

HyperD consists of four integrated components[^src-hyperd-hybrid-periodicity-decoupling]:

1. **[[frequency-aware-residual-representation|Frequency-Aware Residual Representation (FR)]]** — decomposes input via RFFT into frequency bands, generating short-term and long-term periodic embeddings alongside a residual signal.

2. **[[spatial-temporal-attentive-encoder|Spatial-Temporal Attentive Encoder (STAE)]]** — dual-pathway encoder with a short-term encoder (STFE) processing the original signal and a long-term encoder (FR-STFE) processing the residual signal, each guided by its corresponding frequency embedding.

3. **[[demlp-decoder|Decoupled MLP Decoder (fDMLP)]]** — two-stage coarse-to-fine decoder that explicitly extracts and removes trend information via a de-trending MLP before fine-grained prediction.

4. **[[dual-view-alignment-loss|Dual-View Alignment (DVA) Loss]]** — aligns short/long-term representations across time and periodicity views using Soft-DTW and cosine similarity.

## Performance

HyperD achieves state-of-the-art results on all four PeMS benchmarks (PEMS03, 04, 07, 08), outperforming 15 baselines including STAEformer, D2STGNN, FreTS, and PDFormer, with particularly strong advantages at longer forecasting horizons (9-step, 12-step)[^src-hyperd-hybrid-periodicity-decoupling].

## Key Innovation

Unlike prior frequency-based models (FEDformer, FreTS) that treat frequency components uniformly, HyperD's [[hybrid-periodicity-decoupling]] approach assigns different periodic scales to different encoder pathways, enabling each pathway to specialize[^src-hyperd-hybrid-periodicity-decoupling].

## Connections

HyperD builds upon and relates to several lines of frequency-domain and periodicity-aware research:

- **[[source-fedformer|FEDformer]]** (ICML 2022) — a frequency enhanced decomposed transformer that uses Fourier and Wavelet transforms for attention, but treats all frequency components uniformly rather than decoupling them for specialized processing[^src-fedformer].
- **[[source-autoformer|Autoformer]]** (NeurIPS 2021) — introduces decomposition as a core architectural block with auto-correlation mechanism, a predecessor to periodicity-aware design in transformers[^src-autoformer].
- **[[source-dualformer|Dualformer]]** (2026) — a contemporary dual-branch architecture for long-term time-series forecasting that employs hierarchical frequency sampling in the time-frequency domain[^src-dualformer].
- **[[source-timesnet|TimesNet]]** (ICLR 2023) — transforms 1D time series into 2D tensors by capturing multiple periods, offering an alternative approach to modeling multi-scale periodicity[^src-timesnet].

[^src-hyperd-hybrid-periodicity-decoupling]: [[source-hyperd-hybrid-periodicity-decoupling]]
[^src-fedformer]: [[source-fedformer]]
[^src-autoformer]: [[source-autoformer]]
[^src-dualformer]: [[source-dualformer]]
[^src-timesnet]: [[source-timesnet]]
