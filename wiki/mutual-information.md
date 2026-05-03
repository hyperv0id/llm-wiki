---
title: "Mutual Information"
type: concept
tags:
  - information-theory
  - cross-modal-learning
  - multimodal-time-series
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# Mutual Information

Mutual information $I(X;Y)$ measures the amount of information that one random variable contains about another. In multimodal time series, it is used to quantify the relevance of text content to time series data and to guide text filtering[^src-multimodal-ts-anomaly-detection].

## Definition

$$I(X;Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

## Application in Multimodal TS

[[mindts|MindTS]] uses mutual information minimization as the objective for its content condenser — by minimizing $I(Z_{\text{con}}; X_{\text{text}})$ while preserving task-relevant information, redundant text is filtered while informative content is retained[^src-multimodal-ts-anomaly-detection].

## Related

- [[content-condenser-reconstruction]] — MindTS's IB-based text filtering
- [[fine-grained-time-text-semantic-alignment]] — complementary alignment approach

[^src-multimodal-ts-anomaly-detection]: [[source-multimodal-ts-anomaly-detection]]
