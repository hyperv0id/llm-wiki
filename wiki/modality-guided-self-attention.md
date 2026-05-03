---
title: "Modality-Guided Self-Attention"
type: technique
tags:
  - multimodal-time-series
  - attention-mechanism
  - modality-fusion
  - arxiv-2026
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# Modality-Guided Self-Attention

**Modality-Guided Multi-head Self-Attention** 是 [[aurora|Aurora]] 中提出的核心技术，用于将多模态领域知识注入时间序列表示建模[^src-aurora]。

## 动机

传统时间序列自注意力机制仅基于数值时间序列计算注意力权重，无法利用文本或图像中包含的领域特定知识。Aurora 通过模态引导的方式，将多模态领域知识作为注意力计算的引导信号[^src-aurora]。

## 机制

Aurora 首先通过 tokenization、encoding 和 distillation 三个阶段从多模态输入（文本、图像）中提取领域知识表示，然后将这些表示作为引导注入多头自注意力机制中[^src-aurora]。

与 [[multi-modality-refinement|MoST 的 SNR 自适应模态选择]]不同，Aurora 采用注意力引导的方式——多模态知识不直接替换或筛选时间序列特征，而是作为注意力计算的额外上下文来引导时间表示的建模方向。

## 与其他模态融合方法的对比

| 方法 | 模型 | 融合策略 |
|------|------|----------|
| **Modality-Guided Attention** | Aurora | 多模态知识引导自注意力计算 |
| SNR-based Modality Selection | [[most|MoST]] | 估计 SNR 后 Gumbel-Sigmoid 门控 |
| Covariate Homogenization | [[unica|UniCA]] | 投影到统一空间后 Pre/Post-Fusion |
| Cross-view Text Fusion | [[mindts|MindTS]] | 内生文本为 query，外生文本为 key/value |
| Adaptive Frequency Fusion | [[vot|VoT]] | 频域分解后自适应加权融合 |

## 相关页面

- [[aurora]] — Aurora 模型
- [[prototype-guided-flow-matching]] — 解码阶段的流匹配技术
- [[multi-modality-refinement]] — MoST 的模态选择技术
- [[covariate-fusion-module]] — UniCA 的协变量融合模块

[^src-aurora]: [[source-aurora]]
