---
title: "Hierarchical Contrastive Fusion"
type: technique
tags:
  - multimodal
  - contrastive-learning
  - fusion
  - time-series
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Hierarchical Contrastive Fusion

## 定义

分层对比融合是一种多层次多模态特征融合方法，通过监督对比、无监督对比和分布对齐三个层次逐步优化跨模态特征的一致性，最终通过分布相似性度量实现加权融合[^src-mtp]。

## 三个融合层次

### 层次 1：监督对比

对已标注数据，将同一类别的不同模态实例拉近，不同类别的推远。给定 m 个类别，每个实例 (x'i, si) 计算监督损失[^src-mtp]：

$$\mathcal{L}_{SUP} = \sum_{M \in \mathcal{Y}} \frac{1}{|M|} \sum_{x \in M} \sum_{s \in M, x \neq s} [\mathcal{L}(x'_v, s_v) + \mathcal{L}(x'_g, s_g) + \mathcal{L}(x'_t, s_t)]$$

### 层次 2：无监督 InfoNCE 对比

对无标注数据，使用 InfoNCE 损失对齐不同模态的同一实例特征，使语义一致的跨模态对在嵌入空间中接近[^src-mtp]：

$$\mathcal{L}_{UNS} = \frac{1}{3|X|} \sum_{i=1}^{|X|} [\mathcal{L}_v + \mathcal{L}_g + \mathcal{L}_t]$$

每种模态轮流作为锚点，其余两种作为正负样本。

### 层次 3：JS 散度分布对齐

计算任意两模态后验概率分布的 Jensen-Shannon 散度，作为模态间相似性度量[^src-mtp]：

$$\Delta = \frac{1}{3}[JS(p(\cdot|x_v)\|p(\cdot|x_g)) + JS(p(\cdot|x_v)\|p(\cdot|x_t)) + JS(p(\cdot|x_g)\|p(\cdot|x_t))]$$

最终融合特征：$$\hat{x} = (1-\Delta) \sum_m K^m x^m + \Delta \sum_m x^m$$

当模态间高度一致（Δ→0），按可训练权重加权；当模态间高度分歧（Δ→1），等权平均。

## 完整目标函数

$$\mathcal{L} = \alpha \mathcal{L}_{SUP} + \beta \mathcal{L}_{UNS} + \gamma \mathcal{L}_{CE}$$

敏感度分析显示 α=0.1 时最优，表明无监督对比和交叉熵分类在 MTP 中贡献更大。

## 与其他融合方法的对比

| 方法 | 对齐方式 | 自适应融合 | 层级化 |
|------|---------|-----------|--------|
| 简单拼接 | 无 | 否 | 否 |
| 注意力融合 | 隐式 | 是 | 否 |
| InfoNCE 对比 | 显式 | 否 | 单一 |
| **分层对比融合** | 显式（监督+无监督） | 是（JS 散度） | **三级** |

## 相关概念

- [[contrastive-learning]] — 对比学习基础
- [[modality-augmentation]] — 增强模态的来源
- [[mtp]] — 使用该融合方法的完整框架
- [[multimodal-traffic-profiling]] — 应用领域

## 引用

[^src-mtp]: [[source-mtp]]
