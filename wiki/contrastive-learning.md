---
title: "Contrastive Learning"
type: technique
tags:
  - representation-learning
  - self-supervised
  - multimodal-alignment
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# Contrastive Learning

Contrastive learning is a representation learning paradigm that trains encoders to produce similar embeddings for semantically related (positive) pairs and dissimilar embeddings for unrelated (negative) pairs, without requiring explicit labels[^src-multimodal-ts-anomaly-detection].

## In Multimodal Time Series

In the multimodal TS context, contrastive learning is used to align time series and text representations in a shared embedding space. Positive pairs are typically time-text pairs from the same temporal segment; negative pairs come from different segments[^src-multimodal-ts-anomaly-detection].

### InfoNCE Loss

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j^+) / \tau)}{\sum_k \exp(\text{sim}(z_i, z_k) / \tau)}$$

## Applications

- [[fine-grained-time-text-semantic-alignment]] — MindTS's patch-level time-text contrastive alignment
- [[endogenous-text-alignment]] — VoT's decomposed trend/seasonal contrastive learning
- [[multi-level-alignment]] — VoT's multi-level alignment framework using contrastive losses

## Related

- [[mutual-information]] — alternative information-theoretic objective for modality interaction
- [[content-condenser-reconstruction]] — complementary approach using reconstruction instead of contrastive learning

[^src-multimodal-ts-anomaly-detection]: [[source-multimodal-ts-anomaly-detection]]
