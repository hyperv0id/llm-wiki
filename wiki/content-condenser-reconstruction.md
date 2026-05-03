---
title: "Content Condenser Reconstruction"
type: technique
tags:
  - multimodal-time-series
  - information-bottleneck
  - mutual-information
  - cross-modal-reconstruction
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# Content Condenser Reconstruction

**Content Condenser Reconstruction** is a technique introduced in [[mindts|MindTS]] (ICLR 2026) for filtering redundant textual information and enhancing cross-modal interaction between time series and text modalities[^src-multimodal-ts-anomaly-detection].

## Motivation

While text provides complementary information to time series, it often contains redundant content — lengthy details, irrelevant descriptions — that can dilute the contribution of genuinely informative text. Existing multimodal methods perform direct fusion, assuming all text is equally useful[^src-multimodal-ts-anomaly-detection].

Random masking or semantic perturbation (paraphrasing, summarization) used in NLP fails in multimodal time series because it doesn't consider relevance to the time series — high-value text may be randomly discarded while low-value text is retained[^src-multimodal-ts-anomaly-detection].

## Two Components

### 1. Content Condenser

Inspired by the **Information Bottleneck (IB) principle**, the content condenser finds an optimal compressed representation $Z^*_{\text{con}}$ that minimizes mutual information with the original text while preserving reconstruction capability:

$$Z^*_{\text{con}} = \arg\min_{P(Z_{\text{con}}|Z_{\text{text}})} I(Z_{\text{text}}; Z_{\text{con}}) + R(\hat{X}, Z_{\text{con}})$$

Where:
- $I(\cdot;\cdot)$ is mutual information between aligned and condensed text
- $R(\cdot,\cdot)$ is the reconstruction objective[^src-multimodal-ts-anomaly-detection]

#### Implementation

1. **Probability matrix**: An MLP computes $\Psi = [\psi_i]_{i=1}^{N}$ from aligned text $Z_{\text{text}}$
2. **Binary mask**: $F \sim \text{Bernoulli}(\Psi) \in \{0,1\}^N$, where higher $\psi_i$ means more likely to retain
3. **Condensed text**: $Z_{\text{con}} = Z_{\text{text}} \odot F$ (element-wise multiplication)
4. **Gradient propagation**: Straight-through estimator (STE) enables differentiable training[^src-multimodal-ts-anomaly-detection]

#### Loss Functions

**KL divergence loss** (upper bound on mutual information):

$$L_{\text{CC}} = \sum_{i=1}^{N} \left[ \psi_i \log \frac{\psi_i}{\mu} + (1-\psi_i) \log \frac{1-\psi_i}{1-\mu} \right]$$

Where $\mu \in (0,1)$ is a hyperparameter controlling compression strength. The model is robust across $\mu = 0.1$ to $0.9$[^src-multimodal-ts-anomaly-detection].

**Smoothness loss** (prevents discontinuity between adjacent patches):

$$\phi_i = p(\psi_{i+1} - \psi_i)^2, \quad L_{\text{SM}} = \frac{1}{N}\sum_{i=1}^{N-1} \phi_i$$

Without $L_{\text{SM}}$, the model generates incoherent and unstable compressed outputs, leading to performance degradation[^src-multimodal-ts-anomaly-detection].

Total condenser loss: $L_{\text{CL}} = L_{\text{CC}} + L_{\text{SM}}$

### 2. Cross-modal Reconstruction

Rather than reconstructing from the full time series (which contains abundant self-information), MindTS reconstructs from **masked** time series $\tilde{X}$ and condensed text $Z_{\text{con}}$:

$$\hat{X} = \text{Projection}(U_{\text{TT}})$$

$$U_{\text{TT}} = \text{FeedForward}(\tilde{H}_{\text{time}} + \text{CrossAttn}(\tilde{H}_{\text{time}}, Z'_{\text{con}}, Z'_{\text{con}}))$$

Where $Z'_{\text{con}} = \text{MSA}(Z_{\text{con}}, Z_{\text{con}}, Z_{\text{con}})$ is self-attention on condensed text, and $\tilde{H}_{\text{time}}$ is the encoded masked time series[^src-multimodal-ts-anomaly-detection].

**Reconstruction loss**: $L_{\text{Rec}} = \|X - \hat{X}\|^2_F$

The mask ratio is typically set near 50% — higher ratios make reconstruction too challenging, degrading performance[^src-multimodal-ts-anomaly-detection].

## Design Rationale

| Design Choice | Rationale |
|---------------|-----------|
| IB-based compression | Theoretically grounded framework for trading off compression vs. preservation |
| Bernoulli sampling + STE | Differentiable binary masking for end-to-end training |
| Smoothness constraint | Prevents unstable, incoherent compression across adjacent patches |
| Masked time series reconstruction | Forces model to rely on text, strengthening cross-modal dependency |
| Shared-weight time encoder | Ensures consistent representations for masked and unmasked time series |

## Comparison with Related Techniques

| Technique | Model | Goal | Method |
|-----------|-------|------|--------|
| **Content Condenser** | MindTS | Filter redundant text | IB + Bernoulli masking |
| [[historical-in-context-learning|HIC]] | [[vot|VoT]] | Error-informed LLM reasoning | Retrieve corrected historical examples |
| [[multi-modality-refinement|SNR-based Selection]] | [[most|MoST]] | Select high-quality modalities | SNR estimation + Gumbel-Sigmoid |
| [[conditional-attention-pooling|CAP]] | [[unica|UniCA]] | Aggregate covariate info | Conditional attention pooling |
| Random masking (NLP) | Various | Filter text | Random token masking |

The content condenser is unique in using **mutual information minimization** with a **reconstruction objective** to guide filtering, rather than random or heuristic selection[^src-multimodal-ts-anomaly-detection].

## Ablation Evidence

- Removing the content condenser causes **significant** performance degradation, confirming that redundant text negatively impacts the model
- Removing cross-modal reconstruction also degrades performance, showing it enhances cross-modal interaction
- Reversing the order (condenser before alignment) degrades performance — filtering before alignment may discard useful time-relevant information prematurely[^src-multimodal-ts-anomaly-detection]

## Related Pages

- [[mindts]] — the MindTS model
- [[fine-grained-time-text-semantic-alignment]] — upstream alignment technique
- [[multimodal-time-series-anomaly-detection]] — task concept
- [[multi-modality-refinement]] — SNR-based modality selection (MoST)
- [[information-bottleneck-principle]] — theoretical foundation

[^src-multimodal-ts-anomaly-detection]: [[source-multimodal-ts-anomaly-detection]]