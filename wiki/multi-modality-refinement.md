---
title: "Multi-Modality Refinement"
type: technique
tags:
  - multi-modality
  - modality-selection
  - signal-to-noise-ratio
  - spatio-temporal
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# Multi-Modality Refinement

**Multi-Modality Refinement** is a technique introduced in [[most|MoST]] (KDD 2026) for adaptively selecting and filtering available modalities in multi-modality spatio-temporal prediction[^src-most]. It addresses the challenge that cross-city multi-modality data exhibit significant variability in both availability and quality.

## Motivation

Different cities may have different sets of available modalities, and even within a city, the quality of each modality varies by context[^src-most]:
- In dense urban regions, satellite imagery captures structural information but offers limited insight into dynamic traffic patterns
- Human activity data (POIs) can be strong indicators of traffic behavior but may be sparse in low-density areas
- Some sensors may entirely lack certain modalities

Traditional approaches either train separate models per city or restrict to common modalities, both of which limit the utilization of multi-modality information.

## Mechanism

### Modality Encoding

Each available modality is encoded by a dedicated encoder and projected into a unified D-dimensional space[^src-most]:
- **Satellite imagery**: ResNet50 → linear projection
- **POI text**: BERT → linear projection
- **Location**: Normalized latitude/longitude coordinates
- **Time series**: Patch-based tokenization with RoPE positional encoding

Missing modalities are masked out based on a modality state matrix.

### SNR-Based Selection

Inspired by Shannon's second law, each modality is treated as an independent information channel[^src-most]. The channel capacity C is proportional to the signal-to-noise ratio (SNR):

$$C \propto \text{SNR}$$

For each modality, two scores are computed via MLPs:
- **Uncertainty score** U*: estimated noise level
- **Relevance score** R*: task relevance

$$\text{SNR}_* = \frac{R_*}{U_*}$$

Modalities with SNR below a threshold θ are deactivated via a Gumbel-Sigmoid gating mechanism with Straight-Through Estimator (STE) for differentiable training[^src-most]. The modality with the highest SNR is always retained to ensure at least one auxiliary modality is available.

### Contrastive Learning

A contrastive loss guides the selector[^src-most]:
- **Modality-noised** samples: one retained modality is replaced with Gaussian noise
- **Modality-dropped** samples: the selected modality is marked as missing
- The loss encourages noised representations to be close to dropped representations while remaining distinct from clean representations

## Advantages

- **Flexible**: Handles any number of available modalities without architectural changes
- **Noise-resilient**: Automatically suppresses low-quality or irrelevant modalities
- **Efficient**: Only high-SNR modalities contribute to downstream computation
- **Differentiable**: Gumbel-Sigmoid + STE enables end-to-end training

## Related Pages

- [[most]] — the MoST model
- [[multi-modality-guided-spatial-expert]] — spatial expert mechanism in MoST
- [[multimodal-time-series-forecasting]] — broader multimodal TS forecasting
- [[signal-to-noise-ratio-modality-selection]] — SNR-based modality selection concept
- [[content-condenser-reconstruction]] — MindTS's content condenser (different approach to modality quality: IB vs SNR)
- [[mindts]] — MindTS multimodal anomaly detection model

[^src-most]: [[source-most]]