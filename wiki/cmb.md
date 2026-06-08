---
title: "CMB (Channel Mamba Block)"
type: technique
tags:
  - mamba
  - state-space-model
  - diffusion-models
  - time-series
  - channel-modeling
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# CMB (Channel Mamba Block)

**CMB** is a unidirectional Mamba module for inter-channel dependency modeling, proposed in [[ssd-ts|SSD-TS]] (KDD 2025) for diffusion-based time series imputation. It captures multivariate correlations across channels (variates) using Mamba's state space dynamics [^src-ssdts].

## Architecture

CMB operates on the channel dimension by transposing the input, applying Mamba along the channel axis, then transposing back [^src-ssdts]:

```
Input (K × L) ──► Transpose → (L × K) ──► Forward Conv ──► Forward PNM ──► Transpose → (K × L) ──► Output
```

Where **PNM** (Parallel Mamba Block) is the same core SSM-based computation unit used in [[bam|BAM]] [^src-ssdts].

## Key Design Rationale

The authors argue that Mamba's **iterative hidden state evolution** $h_t = A_t h_{t-1} + B_t x_t$ is more suitable for inter-channel dependency modeling than three alternatives [^src-ssdts]:

1. **Attention-based channel modeling**: Relies on static attention matrices derived from input similarity, failing to describe dynamic, evolving cross-channel dependencies. Mamba's iterative state update provides an adaptive mechanism that tracks how channel relationships evolve over time.

2. **Convolution-based channel modeling**: Fixed kernel assumptions cannot capture varying local dependencies, and limited receptive fields miss global inter-channel interactions.

3. **SENet-style Channel Attention**: Ablation shows replacing CMB with channel attention (Squeeze-and-Excitation) causes a significant performance drop, proving SSM-based channel modeling is superior [^src-ssdts].

## Dynamic Dependencies

A key insight of CMB: even for the same time series, the degree of interaction between channels varies depending on the specific downstream task. Mamba's input-dependent parameters ($\Delta_i$, $B_i$, $C_i$) naturally adapt to this variability, unlike fixed attention or convolution weights [^src-ssdts].

## Ablation Impact

On MuJoCo with 90% missing rate [^src-ssdts]:

| Configuration | MSE |
|--------------|-----|
| BAM + CMB (full) | $5.46\times10^{-4}$ |
| BAM only (no CMB) | $7.48\times10^{-4}$ (+37%) |
| BAM + Channel Attention | $7.43\times10^{-4}$ (+36%) |

Removing CMB causes a notable performance degradation. Replacing CMB with channel attention yields similar degradation — confirming that **Mamba-based channel modeling** is more effective than attention-based alternatives [^src-ssdts].

## Comparison with Related Approaches

| Method | Inter-Channel Modeling | Mechanism |
|--------|----------------------|-----------|
| CSDI | Feature Transformer | Self-attention across K channels at each time step |
| SSSD | S4 SSM | S4 state space model on channel dimension |
| S-Mamba | Bidirectional Mamba VC | Bidirectional Mamba scanning all variate tokens |
| **CMB (SSD-TS)** | **Unidirectional Mamba** | **Transposed Mamba scan across channels within diffusion denoiser** |

## Related Pages

- [[ssd-ts|SSD-TS]] — the parent model
- [[bam|BAM (Bidirectional Attention Mamba)]] — the intra-channel counterpart
- [[mamba|Mamba]] — the underlying state space model
- [[s-mamba|S-Mamba]] — bidirectional Mamba for cross-variable correlation in forecasting
- [[cross-dimension-dependency|Cross-Dimension Dependency]] — general concept of modeling inter-variate relationships

[^src-ssdts]: [[source-ssdts]]
