---
title: "Contrastive Learning"
type: technique
tags:
  - representation-learning
  - self-supervised
  - multimodal-alignment
created: 2026-05-03
last_updated: 2026-06-08
source_count: 2
confidence: medium
status: active
---

# Contrastive Learning

Contrastive learning is a representation learning paradigm that trains encoders to produce similar embeddings for semantically related (positive) pairs and dissimilar embeddings for unrelated (negative) pairs, without requiring explicit labels[^src-multimodal-ts-anomaly-detection].

## In Multimodal Time Series

In the multimodal TS context, contrastive learning is used to align time series and text representations in a shared embedding space. Positive pairs are typically time-text pairs from the same temporal segment; negative pairs come from different segments[^src-multimodal-ts-anomaly-detection].

### InfoNCE Loss

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j^+) / \tau)}{\sum_k \exp(\text{sim}(z_i, z_k) / \tau)}$$

## In Imputation: Mask-Invariant Representations

[[nuwats|NuwaTS]] (arXiv 2024) applies contrastive learning to a different axis — robustness to **missing patterns** rather than modality alignment[^src-nuwats]. For each input, it generates two views with **different mask ratios** and feeds both through the PLM. The representation of the *same patch* under the two masks forms a positive pair, while representations from other patches and other series are negatives. NuwaTS uses an InfoNCE objective with a **bi-linear inner product** $q^\top W k_+$ (learnable $W$) combined with MSE reconstruction loss[^src-nuwats]. The effect is **mask-invariant patch embeddings**: the model learns to produce consistent representations regardless of how much of a patch is missing, which ablations show is necessary for cross-domain zero-shot imputation[^src-nuwats].

## Applications

- [[fine-grained-time-text-semantic-alignment]] — MindTS's patch-level time-text contrastive alignment
- [[endogenous-text-alignment]] — VoT's decomposed trend/seasonal contrastive learning
- [[multi-level-alignment]] — VoT's multi-level alignment framework using contrastive losses
- [[nuwats]] — NuwaTS's mask-invariant patch representations across missing patterns

## Related

- [[mutual-information]] — alternative information-theoretic objective for modality interaction
- [[content-condenser-reconstruction]] — complementary approach using reconstruction instead of contrastive learning

[^src-multimodal-ts-anomaly-detection]: [[source-multimodal-ts-anomaly-detection]]
[^src-nuwats]: [[source-nuwats]]
