---
title: "Information Bottleneck Principle"
type: concept
tags:
  - information-theory
  - representation-compression
  - multimodal-time-series
created: 2026-05-03
last_updated: 2026-11-24
source_count: 2
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

## 在长程时序预测窗口滤波中的同构应用

[[source-pih|PIH]] 将信息瓶颈上界直接用于长回看窗口特征去噪（同构实现）：在 Patch Embedding 之后用 `MLP + Sigmoid` 与 `Gumbel-Sigmoid` 采样生成 patch 级伯努利掩码，按掩码注入同均值方差高斯噪声抑制冗余历史，并以预测 MSE 联合变分压缩上界优化，避免对长历史子序列做组合枚举；超参灵敏度表明压缩权重 $\beta$ 与 Gumbel 温度 $\tau$ 直接左右滤波强度，但正文消融仅有图示而无组件级量化数值表[^src-pih]。

## Related

- [[content-condenser-reconstruction]] — MindTS's IB-based text filtering implementation
- [[source-pih]] — 序列版 `Gumbel-Sigmoid` 噪声掩码变分上界在长程预测中的同构实现
- [[mutual-information]] — the measure used to quantify the IB compression
- [[elbo]] — ELBO as a special case of IB in variational inference

[^src-multimodal-ts-anomaly-detection]: [[source-multimodal-ts-anomaly-detection]]
[^src-pih]: [[source-pih]]
