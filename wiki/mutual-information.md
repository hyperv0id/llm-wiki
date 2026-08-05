---
title: "Mutual Information"
type: concept
tags:
  - information-theory
  - cross-modal-learning
  - multimodal-time-series
created: 2026-05-03
last_updated: 2026-08-05
source_count: 2
confidence: medium
status: active
---

# Mutual Information

Mutual information $I(X;Y)$ measures the amount of information that one random variable contains about another. In multimodal time series, it is used to quantify the relevance of text content to time series data and to guide text filtering[^src-multimodal-ts-anomaly-detection].

## Definition

$$I(X;Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

## Application in Multimodal TS

[[mindts|MindTS]] uses mutual information minimization as the objective for its content condenser — by minimizing $I(Z_{\text{con}}; X_{\text{text}})$ while preserving task-relevant information, redundant text is filtered while informative content is retained[^src-multimodal-ts-anomaly-detection].

## 双向使用：MI Minimax 解耦

[[midas|MIDAS]]（不完全多模态情感分析，TPAMI 2026）把互信息以 minimax 形式双向使用[^src-midas]：最小化同一模态共享与独有潜因子之间的 $I(Z^s_m; Z^e_m)$ 实现解耦（经 interaction information 分解 + 变分上界可优化），最大化跨模态共享空间之间的 $I(Z^s_{m_1}; Z^s_{m_2})$ 实现语义对齐（Deep InfoMax 的 JSD 估计器 + 硬负样本）。与 MindTS 的单向压缩不同，MIDAS 用互信息同时扮演"去噪"与"对齐"两个角色，详见 [[mutual-information-disentanglement]]。

## Related

- [[content-condenser-reconstruction]] — MindTS's IB-based text filtering
- [[fine-grained-time-text-semantic-alignment]] — complementary alignment approach
- [[mutual-information-disentanglement]] — MIDAS 的 MI minimax 解耦技术
- [[midas]] — 不完全多模态情感分析框架

[^src-multimodal-ts-anomaly-detection]: [[source-multimodal-ts-anomaly-detection]]
[^src-midas]: [[source-midas]]
