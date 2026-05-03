---
title: "Information Bottleneck Principle"
type: concept
tags:
  - information-theory
  - representation-compression
  - multimodal-time-series
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# Information Bottleneck Principle

The Information Bottleneck (IB) principle, introduced by Tishby et al. (2000), provides a framework for learning compressed representations that retain maximal information about a target variable while discarding irrelevant details. Formally, a representation $T$ of input $X$ should minimize $I(T;X)$ (compression) while maximizing $I(T;Y)$ (prediction)[^src-multimodal-ts-anomaly-detection].

## IB Objective

$$\min_{p(t|x)} \mathcal{L}_{\text{IB}} = I(T;X) - \beta \cdot I(T;Y)$$

where $\beta$ controls the trade-off between compression and prediction.

## Application in Multimodal TS

[[mindts|MindTS]] applies the IB principle to filter redundant text in multimodal anomaly detection. The content condenser minimizes mutual information between the compressed text representation $Z_{\text{con}}$ and the raw text $X_{\text{text}}$, while using a reconstruction objective to ensure task-relevant information is preserved[^src-multimodal-ts-anomaly-detection].

## Related

- [[content-condenser-reconstruction]] — MindTS's IB-based text filtering implementation
- [[mutual-information]] — the measure used to quantify the IB compression
- [[elbo]] — ELBO as a special case of IB in variational inference

[^src-multimodal-ts-anomaly-detection]: [[source-multimodal-ts-anomaly-detection]]
