---
title: "Fleximodal Fusion"
type: technique
tags:
  - multimodal-fusion
  - missing-modality
  - robustness
created: 2026-07-21
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

# Fleximodal Fusion

**Fleximodal Fusion** 是 [[omnifield|OmniField]] 中处理任意输入模态子集的技术。通过模态存在掩码（presence mask）使单一模型在训练和推理时适应任意可用模态组合，无需插补缺失通道[^src-omnifield]。

## 机制

令 $\mathcal{M} = \{1, \dots, M\}$ 为模态索引，$\boldsymbol{\pi} \in \{0,1\}^M$ 指示可用模态（$\pi_m = 1$ 当且仅当 $N_m > 0$）[^src-omnifield]：

1. **门控编码器**：$\tilde{Z}_m = \pi_m \cdot \mathcal{E}_m(X_m)$，缺失通道输出零向量
2. **掩码注意力**：cross-attention 中对缺失模态的 key/value 施加 $-\infty$ 偏置，等效移除
3. **Flexi-objective**：$\mathcal{L} = \sum_{m=1}^{M} \tau_m \cdot \ell(\hat{Y}_m, Y_m)$，仅在有监督目标的模态上计算损失

## 与 ModDrop 的区别

[[fleximodal-fusion|Fleximodal Fusion]] 本质上不同于 ModDrop（Neverova et al., 2015）的训练时随机丢弃策略：它处理**真正缺失的通道**（如 EPA-AQS 中某些天某些污染物站点无记录），而非仅作为训练增强[^src-omnifield]。门控发生在编码器源端、注意力层和损失函数三个层面，防止缺失通道产生"幻觉"证据[^src-omnifield]。

## 相关

- [[omnifield]] — 使用 Fleximodal Fusion 的完整模型
- [[multimodal-crosstalk]] — 在同一架构中处理实际可用的模态
- [[iterative-cross-modal-refinement]] — 在可用模态子集上进行迭代精炼

[^src-omnifield]: [[source-omnifield]]
