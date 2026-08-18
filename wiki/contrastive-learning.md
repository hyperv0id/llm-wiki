---
title: "Contrastive Learning"
type: technique
tags:
  - representation-learning
  - self-supervised
  - multimodal-alignment
created: 2026-05-03
last_updated: 2026-08-19
source_count: 7
confidence: medium
status: active
---

# Contrastive Learning

Contrastive learning is a representation learning paradigm that trains encoders to produce similar embeddings for semantically related (positive) pairs and dissimilar embeddings for unrelated (negative) pairs, without requiring explicit labels[^src-multimodal-ts-anomaly-detection].

## In Multimodal Time Series

In the multimodal TS context, contrastive learning is used to align time series and text representations in a shared embedding space. Positive pairs are typically time-text pairs from the same temporal segment; negative pairs come from different segments[^src-multimodal-ts-anomaly-detection].

### Trimodal limits (TS · vision · language)

[[ts-vl-alignment|Yashwante & Yu (2026)]] probe CLIP-style **post-hoc** contrastive alignment with **frozen** pretrained encoders and shared projection heads over time series, plots, and text. Independently pretrained spaces are **near-orthogonal** without coupling; after InfoNCE, alignment is **asymmetric** (TS–IMG stronger than TS–TXT), improves unevenly with scale, and **saturates** with caption information density—so contrastive projection alone does not guarantee fine-grained multimodal convergence for numeric series[^src-ts-vl-alignment].

### Cross-modal misalignment under MMCL

[[cross-modal-misalignment|Cai, Liu et al. (NeurIPS 2025)]] formalize image–text **selection** (omitted semantics) and **perturbation** (altered semantics) biases in the generative process. Under the asymptotic alignment–entropy objective of multimodal contrastive learning, encoders **block-identify only the unbiased shared semantic subset**; omitted/perturbed factors and modality-specific noise are excluded—so misalignment is a semantic filter (mitigate for full-coverage pretraining; leverage when it matches environment-sensitive factors for OOD)[^src-cross-modal-misalignment].

### InfoNCE Loss

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j^+) / \tau)}{\sum_k \exp(\text{sim}(z_i, z_k) / \tau)}$$

## In Imputation: Mask-Invariant Representations

[[nuwats|NuwaTS]] (arXiv 2024) applies contrastive learning to a different axis — robustness to **missing patterns** rather than modality alignment[^src-nuwats]. For each input, it generates two views with **different mask ratios** and feeds both through the PLM. The representation of the *same patch* under the two masks forms a positive pair, while representations from other patches and other series are negatives. NuwaTS uses an InfoNCE objective with a **bi-linear inner product** $q^\top W k_+$ (learnable $W$) combined with MSE reconstruction loss[^src-nuwats]. The effect is **mask-invariant patch embeddings**: the model learns to produce consistent representations regardless of how much of a patch is missing, which ablations show is necessary for cross-domain zero-shot imputation[^src-nuwats].

## Dual-level Hard Negative Mining in TRACE

[[trace|TRACE]]（NeurIPS 2025）在跨模态对齐中提出 [[dual-level-hard-negative-mining|双级硬负采样]]——在 sample-level（[CLS] 嵌入 vs 样本文本）和 channel-level（[[channel-identity-token|CIT]] 嵌入 vs 通道文本）两个粒度上动态挖掘硬负样本。channel-level 引入 intra-instance（同实例其他通道）和 inter-instance（跨实例同通道）两类 distractor，使模型区分"看起来相似但语义不同"的通道模式[^src-trace-neurips2025]。这与传统 CLIP 式仅做 sample-level 随机负采样不同，因为文本常引用具体变量（如温度峰值、风速），单全局嵌入无法精确对齐[^src-trace-neurips2025]。

## Applications

- [[fine-grained-time-text-semantic-alignment]] — MindTS's patch-level time-text contrastive alignment
- [[endogenous-text-alignment]] — VoT's decomposed trend/seasonal contrastive learning
- [[multi-level-alignment]] — VoT's multi-level alignment framework using contrastive losses
- [[nuwats]] — NuwaTS's mask-invariant patch representations across missing patterns
- [[ts-vl-alignment]] — limits of post-hoc contrastive alignment across time series, vision, and language
- [[cross-modal-misalignment]] — selection/perturbation bias; what MMCL retains under misaligned pairs
- [[trace]] — TRACE dual-level hard negative mining for cross-modal TS-text retrieval

## Related

- [[mutual-information]] — alternative information-theoretic objective for modality interaction
- [[content-condenser-reconstruction]] — complementary approach using reconstruction instead of contrastive learning
- [[fine-grained-traffic-prediction]] — fine-grained traffic prediction, where contrastive clustering is used for efficient graph partitioning
- [[ssdl]] — ST-SSDL 使用 prototype triplet loss 实现潜在空间离散化，作为自监督偏差学习的基础
- [[source-cross-modal-misalignment]] — NeurIPS 2025 theory of misalignment value in MMCL

[^src-st-ssdl]: [[source-st-ssdl]]

[^src-multimodal-ts-anomaly-detection]: [[source-multimodal-ts-anomaly-detection]]
[^src-minitraffic]: [[source-minitraffic]]
[^src-nuwats]: [[source-nuwats]]
[^src-ts-vl-alignment]: [[source-ts-vl-alignment]]
[^src-cross-modal-misalignment]: [[source-cross-modal-misalignment]]
[^src-trace-neurips2025]: [[source-trace-neurips2025]]
