---
title: "Aurora"
type: entity
tags:
  - multimodal-time-series
  - foundation-model
  - generative-forecasting
  - flow-matching
  - zero-shot
  - arxiv-2026
created: 2026-05-03
last_updated: 2026-05-03
source_count: 4
confidence: high
status: active
---

# Aurora

**Aurora** 是首个多模态时间序列基础模型（Multimodal Time Series Foundation Model），由 Wu, Jin, Qiu, Chen, Shu, Yang 和 Guo 提出（arXiv:2509.22295, 2026）[^src-aurora]。Aurora 支持多模态输入（文本、图像、数值时间序列）和零样本推理，通过生成式概率预测实现跨域泛化。

## 核心能力

Aurora 填补了现有时间序列基础模型的两个关键空白[^src-aurora]：

1. **单模态 TSFMs**（如 [[timesfm|TimesFM]]、[[chronos|Chronos]]）缺乏对领域特定知识（文本、图像）的显式利用
2. **端到端多模态监督模型**不支持零样本推理

Aurora 在 Cross-domain Multimodal Time Series Corpus 上预训练，能够自适应提取和聚焦于文本或图像模态中包含的关键领域知识[^src-aurora]。

## 架构

### 编码阶段

1. **Tokenization + Encoding + Distillation**：从多模态输入中提取领域知识作为引导
2. **Modality-Guided Multi-head Self-Attention**：将多模态领域知识注入时间表示建模[^src-aurora]

### 解码阶段

**Prototype-Guided Flow Matching**：多模态表示用于生成未来 token 的条件和原型，实现生成式概率预测[^src-aurora]。

## 与现有模型的对比

| 维度 | Aurora | [[simdiff|SimDiff]] | [[most|MoST]] | [[timesfm|TimesFM]] | [[chronos|Chronos]] |
|------|--------|---------------------|---------------|---------------------|---------------------|
| 类型 | 多模态基础模型 | 单模态扩散模型 | 多模态 ST 基础模型 | 单模态基础模型 | 单模态基础模型 |
| 模态 | 文本 + 图像 + 数值 | 仅数值 | 图像 + 文本 + 位置 + TS | 仅数值 | 仅数值 |
| 生成方式 | Flow Matching | Diffusion (DDPM) | 判别式 | 自回归 | 自回归 |
| 零样本 | ✓ | ✗ | ✓ | ✓ | ✓ |
| 跨域泛化 | ✓ | ✗ | ✓ (跨城市) | ✓ (跨数据集) | ✓ (跨数据集) |
| 概率预测 | ✓ (生成式) | ✓ (扩散式) | ✗ | ✗ | ✗ |

### 与 SimDiff 的对比

两者都是生成式方法，但：
- **SimDiff** 使用扩散模型（DDPM）进行点预测，仅支持单模态数值输入[^src-simdiff]
- **Aurora** 使用 Flow Matching 进行概率预测，支持多模态输入和零样本推理[^src-aurora]

### 与 MoST 的对比

两者都是多模态基础模型，但：
- **MoST** 是判别式模型，通过 SNR 自适应模态选择进行时空交通预测[^src-most]
- **Aurora** 是生成式模型，通过 Modality-Guided Attention 和 Prototype-Guided Flow Matching 进行通用时间序列预测[^src-aurora]

### 与 VoT 的对比

两者都利用多模态文本信息，但：
- **[[vot|VoT]]** 使用 LLM 进行事件驱动推理，通过多级对齐融合文本与时间序列[^src-event-driven-ts-forecasting]
- **Aurora** 通过 tokenization-encoding-distillation 提取领域知识，使用注意力引导注入时间建模[^src-aurora]

## 实验

在 5 个基准上评估：TimeMMD、TSFM-Bench、ProbTS、TFB 和 EPF，单模态和多模态场景均 SOTA[^src-aurora]。

## 相关页面

- [[source-aurora]] — 源文件摘要
- [[modality-guided-self-attention]] — 模态引导自注意力技术
- [[prototype-guided-flow-matching]] — 原型引导流匹配技术
- [[generative-time-series-forecasting]] — 生成式时间序列预测概念
- [[multimodal-time-series-forecasting]] — 多模态时间序列预测概念
- [[flow-matching]] — Flow Matching 理论基础
- [[simdiff]] — 扩散式生成预测对比
- [[most]] — 判别式多模态基础模型对比
- [[vot]] — LLM 推理式多模态预测对比
- [[tats]] — TaTS 即插即用多模态框架（Aurora 为生成式基础模型，TaTS 为轻量级插件）
- [[timesfm]] — 单模态 TSFM 对比
- [[chronos]] — 单模态 TSFM 对比
- [[uniextreme]] — UniExtreme 极端天气基础模型（Aurora: 通用 TS 多模态生成式；UniExtreme: 天气领域极端事件判别式）

[^src-aurora]: [[source-aurora]]
[^src-simdiff]: [[source-simdiff]]
[^src-most]: [[source-most]]
[^src-event-driven-ts-forecasting]: [[source-event-driven-ts-forecasting]]
