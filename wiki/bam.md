---
title: "BAM (Bidirectional Attention Mamba)"
type: technique
tags:
  - mamba
  - state-space-model
  - diffusion-models
  - time-series
  - attention
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# BAM (Bidirectional Attention Mamba)

**BAM** is a bidirectional Mamba module with integrated temporal attention, proposed in [[ssd-ts|SSD-TS]] (KDD 2025) for intra-channel, multi-range dependency modeling in diffusion-based time series imputation [^src-ssdts].

## Architecture

BAM operates on individual channels (variates) of a multivariate time series, processing the temporal dimension bidirectionally [^src-ssdts]:

```
Input (1 × L) ──► Forward Conv ──► Forward PNM ──┐
                ──► Backward Conv ──► Backward PNM ──┤
                                                     ▼
                                            Temporal Attention
                                                     ▼
                                            Output (1 × L)
```

Where **PNM** (Parallel Mamba Block) is the core SSM-based computation unit with forward/backward convolution + SSM scans + gating + normalization + residual connections [^src-ssdts].

## Key Design Decisions

1. **Bidirectional Scanning**: Unlike standard unidirectional Mamba ($h_t$ depends only on $h_{t-1}$), BAM runs both forward and backward Mamba scans in parallel. This ensures each time step captures context from both directions — critical for imputation where future and past observations both inform missing values [^src-ssdts].

2. **Temporal Attention**: After bidirectional Mamba processing, a temporal self-attention layer is applied to capture explicit multi-range dependencies. Ablation shows temporal attention is the **most impactful** component — its removal causes the largest performance drop [^src-ssdts].

3. **Gating Mechanism**: Mamba's inherent gating (via $\Delta_i$ input gate and $\widetilde{A}_i$ forget gate) helps the module adaptively filter noise during different diffusion stages, a key advantage over fixed attention weights in early diffusion steps [^src-ssdts].

## Why Bidirectional + Attention?

- Bidirectional Mamba captures **sequential intra-channel** patterns (trend, seasonality) with linear complexity
- Temporal attention captures **explicit multi-range** dependencies (e.g., daily periodicity at lag 24) that Mamba's sequential scanning may dilute
- Together they provide complementary views of the same temporal sequence [^src-ssdts]

## Ablation Impact

On MuJoCo with 90% missing rate, removing BAM's bidirectional design and reverting to unidirectional form causes performance degradation, confirming bidirectional modeling captures more intra-channel dependency [^src-ssdts].

## Comparison with Related Approaches

| Approach | Temporal Modeling | Complexity |
|----------|------------------|------------|
| CSDI's Time Transformer | Self-attention over all time steps | $O(L^2)$ |
| SSSD's S4 | Bidirectional S4 state space model | $O(L)$ |
| **BAM (SSD-TS)** | **Bidirectional Mamba + temporal attention** | $O(L)$ (Mamba) + $O(L^2)$ (attention, but typically small L) |

BAM achieves linear Mamba complexity with the expressive power of attention for intra-channel modeling [^src-ssdts].

## Related Pages

- [[ssd-ts|SSD-TS]] — the parent model
- [[cmb|CMB (Channel Mamba Block)]] — the inter-channel counterpart
- [[mamba|Mamba]] — the underlying state space model
- [[s-mamba|S-Mamba]] — bidirectional Mamba for forecasting (different task, earlier work)

[^src-ssdts]: [[source-ssdts]]
