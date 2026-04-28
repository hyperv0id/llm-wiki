---
title: "Frequency-Aware Residual Representation"
type: technique
tags:
  - fourier-transform
  - signal-decomposition
  - traffic-forecasting
created: 2026-04-27
last_updated: 2026-04-28
source_count: 4
confidence: high
status: active
---

# Frequency-Aware Residual Representation (FR)

The Frequency-Aware Residual Representation (FR) module is a component of [[hyperd|HyperD]] that decomposes historical traffic data into periodic and residual components via Fourier analysis[^src-hyperd-hybrid-periodicity-decoupling].

## Process

1. **RFFT Transform** — Apply Real Fast Fourier Transform along the temporal dimension to convert input X ∈ ℝ^{T×N×C} into frequency domain F ∈ ℂ^{K×N×C}, where K = ⌊T/2⌋ + 1[^src-hyperd-hybrid-periodicity-decoupling].

2. **Frequency Band Clustering** — Partition frequency components into three bands using learned thresholds[^src-hyperd-hybrid-periodicity-decoupling]:
   - **High-frequency band** → Residual component (rapid fluctuations, noise)
   - **Mid-frequency band** → Short-term periodic embedding (H_s)
   - **Low-frequency band** → Long-term periodic embedding (H_l)

3. **Residual Reconstruction** — Apply inverse RFFT (IRFFT) to high-frequency components to obtain the residual signal R in the time domain[^src-hyperd-hybrid-periodicity-decoupling].

4. **Embedding Generation** — Learnable embeddings H_s and H_l are produced from mid- and low-frequency bands to guide the respective encoder pathways in [[spatial-temporal-attentive-encoder|STAE]][^src-hyperd-hybrid-periodicity-decoupling].

## Design Principle

Unlike FreTS, which applies graph convolutions to all frequency components uniformly, FR explicitly **separates** frequency bands and generates distinct embeddings. This enables the downstream [[hybrid-periodicity-decoupling|dual-pathway encoder]] to process short-term and long-term periodicity with specialized architectures[^src-hyperd-hybrid-periodicity-decoupling].

## Impact

Ablation studies show removing the FR module causes the largest performance drop (+3.2% MAE) among all HyperD components, confirming it is the most critical architectural element[^src-hyperd-hybrid-periodicity-decoupling].

## Learned Cutoffs

The learned frequency thresholds align with interpretable real-world periods: daily cycle at 288 steps/day (5-min sampling), weekly cycle at 2016 steps/week[^src-hyperd-hybrid-periodicity-decoupling].

## Related Frequency-Domain Methods

The FR module's frequency-band separation principle relates to other frequency-domain techniques:

- **[[source-fedformer|FEDformer]]** (ICML 2022) uses Fourier-enhanced and Wavelet-enhanced attention blocks to capture frequency patterns, though it processes all frequency components uniformly without explicit band separation[^src-fedformer].
- **[[source-frets|FreTS]]** (NeurIPS 2023) applies MLPs in the frequency domain to all components uniformly, in contrast to FR's targeted embedding per frequency band[^src-frets].
- **[[source-afe-tfnet|AFE-TFNet]]** combines wavelet transform (WT) and FFT for adaptive multi-scale feature extraction, sharing FR's motivation of multi-resolution frequency analysis for time-series modeling[^src-afe-tfnet].

[^src-hyperd-hybrid-periodicity-decoupling]: [[source-hyperd-hybrid-periodicity-decoupling]]
[^src-fedformer]: [[source-fedformer]]
[^src-frets]: [[source-frets]]
[^src-afe-tfnet]: [[source-afe-tfnet]]
