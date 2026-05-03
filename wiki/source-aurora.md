---
title: "Aurora: Towards Universal Generative Multimodal Time Series Forecasting"
type: source-summary
tags:
  - multimodal-time-series
  - foundation-model
  - generative-forecasting
  - flow-matching
  - zero-shot
  - arxiv-2026
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
source_count: 1
confidence: medium
status: active
---

# Aurora: Towards Universal Generative Multimodal Time Series Forecasting

**Wu, Jin, Qiu, Chen, Shu, Yang & Guo (2026), arXiv:2509.22295**

## 核心论题

Aurora 提出首个多模态时间序列基础模型（Multimodal Time Series Foundation Model），支持多模态输入（文本、图像、数值）和零样本推理，通过生成式概率预测实现跨域泛化[^src-aurora]。核心论点：现有单模态 TSFMs（如 TimesFM、Chronos）缺乏对领域特定知识（文本、图像）的显式利用，而端到端多模态监督模型不支持零样本推理——Aurora 填补了这一空白[^src-aurora]。

## 方法

### 多模态领域知识提取

Aurora 通过 tokenization、encoding 和 distillation 三个阶段从文本或图像模态中提取领域知识作为引导[^src-aurora]：

1. **Tokenization**：将多模态输入转换为统一 token 表示
2. **Encoding**：编码多模态 token 为领域知识表示
3. **Distillation**：蒸馏关键领域知识，过滤冗余信息

### Modality-Guided Multi-head Self-Attention

将提取的多模态领域知识注入时间表示建模中，通过模态引导的多头自注意力机制实现[^src-aurora]。与 MoST 的 SNR 自适应模态选择不同，Aurora 采用注意力引导的方式将多模态信息融入时间建模。

### Prototype-Guided Flow Matching

在解码阶段，多模态表示用于生成未来 token 的条件和原型（prototypes），贡献了一种新颖的**原型引导流匹配**（Prototype-Guided Flow Matching）用于生成式概率预测[^src-aurora]。与标准 Flow Matching 不同，Aurora 使用多模态表示生成的条件和原型来引导流匹配过程。

## 实验

在 5 个公认基准上评估：TimeMMD、TSFM-Bench、ProbTS、TFB 和 EPF，在单模态和多模态场景下均取得一致的 SOTA 性能[^src-aurora]。

## 关键贡献

1. 首个支持多模态输入和零样本推理的多模态时间序列基础模型
2. Modality-Guided Multi-head Self-Attention 将多模态领域知识注入时间建模
3. Prototype-Guided Flow Matching 实现生成式概率预测蛐
4. 在 5 个基准上单模态和多模态场景均 SOTA

## 局限性

- 仅基于 arXiv 摘要分析，完整论文细节待补充
- 预训练语料库（Cross-domain Multimodal Time Series Corpus）的具体构成未详述

[^src-aurora]: [[source-aurora]]
