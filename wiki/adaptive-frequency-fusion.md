---
title: "Adaptive Frequency Fusion (AFF)"
type: technique
tags:
  - time-series
  - frequency-domain
  - multimodal
  - fusion
  - fft
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# Adaptive Frequency Fusion (AFF)

**Adaptive Frequency Fusion (AFF)** is a prediction-level alignment technique introduced in [[vot|VoT]] (ICLR 2026) that dynamically fuses the frequency components of event-driven predictions and numerical predictions to achieve complementary advantages across modalities[^src-event-driven-ts-forecasting].

## Motivation

Event-driven predictions (from exogenous text reasoning) excel at capturing patterns influenced by external factors, while numerical predictions (from time series modeling) capture intrinsic temporal patterns. These complementary strengths can be integrated through frequency-based approaches — different frequency bands may benefit differently from each modality[^src-event-driven-ts-forecasting].

Rather than assuming fixed influence of exogenous text on time series, AFF learns optimal fusion strategies directly from the dataset.

## Mechanism

### 1. Frequency Decomposition

Both branch predictions are decomposed into frequency components via FFT:

$$\mathcal{F}^{\text{num}} = \text{FFT}(\hat{\mathbf{Y}}^{\text{num}}), \quad \mathcal{F}^{\text{event}} = \text{FFT}(\hat{\mathbf{Y}}^{\text{event}})$$

### 2. Band Partitioning

The spectrum is partitioned into three bands (low, mid, high) based on frequency. Band-specific components are extracted with masks:

$$\mathcal{F}_{*}^{b} = \mathcal{F}_{*} \odot \mathbf{M}^{b}, \quad *\in\{\text{num},\text{event}\}, \quad b\in\{\text{low},\text{mid},\text{high}\}$$

### 3. Adaptive Weighted Fusion

Learnable weights $\mathbf{w} = \{w_{*}^{b}\}$ adapt to data characteristics:

$$\mathcal{F}_{\text{fused}} = \sum_{*}\sum_{b} w_{*}^{b} \mathcal{F}_{*}^{b}, \quad \hat{\mathbf{Y}}_{\text{final}} = \text{iFFT}(\mathcal{F}_{\text{fused}})$$

The weights are learned per frequency band and per branch, enabling domain-driven optimization — different datasets have varying dependencies on textual vs. numerical information across frequency bands[^src-event-driven-ts-forecasting].

## Design Rationale

| Design Choice | Rationale |
|---------------|-----------|
| Frequency-domain fusion | Different frequency components carry different types of information (trends in low freq, events in mid/high freq) |
| Learnable weights (not fixed) | Datasets vary in how much they depend on text vs. numerical patterns |
| Three-band partition | Balances granularity with parameter efficiency |
| FFT + iFFT | Enables clean separation and recombination of frequency components |

## Comparison with Related Fusion Techniques

| Technique | Model | Fusion Domain | Adaptive? |
|-----------|-------|---------------|-----------|
| **AFF** | VoT | Frequency | Yes (learnable weights) |
| [[covariate-fusion-module|Dual-stage Fusion]] | [[unica|UniCA]] | Time | Yes (attention-based) |
| [[multi-modality-refinement|SNR-based Selection]] | [[most|MoST]] | Modality | Yes (SNR + Gumbel-Sigmoid) |
| Cross-view Attention | [[mindts|MindTS]] | Representation | Yes (attention) |

AFF is unique in performing fusion in the **frequency domain** with **learnable per-band weights**, rather than in the time domain or at the modality level[^src-event-driven-ts-forecasting].

## Related Pages

- [[vot]] — the VoT model
- [[multi-level-alignment]] — the overarching alignment concept
- [[endogenous-text-alignment]] — representation-level alignment (ETA)
- [[event-driven-reasoning]] — the event-driven prediction branch
- [[source-event-driven-ts-forecasting]] — source summary

[^src-event-driven-ts-forecasting]: [[source-event-driven-ts-forecasting]]
