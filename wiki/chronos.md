---
title: "Chronos"
type: entity
tags:
  - time-series
  - foundation-model
  - tokenizer
  - iclr2024
created: 2026-04-29
last_updated: 2026-04-29
source_count: 2
confidence: high
status: active
---

# Chronos

## 概述

**Chronos** 是 Amazon 提出的预训练时间序列语言模型，发表于 ICLR 2024[^src-chronos]。

## 核心贡献

1. **时间序列分词器 (Tokenizer)**：将连续时间序列量化为离散 token
2. **语言模型架构**：类似 T5 的编码器-解码器结构
3. **大规模预训练**：在多种领域的时间序列数据上训练
4. **零样本和少样本预测**：无需微调即可预测

## 架构

### Tokenizer

将时间序列通过以下步骤转换为离散 token：
1. **量化**：使用分位数将连续值离散化
2. **Binning**：将时间序列映射到固定数量的 bin
3. **Embedding**：将 bin ID 转换为向量

### 模型

- **架构**：类似 T5 的编码器-解码器 Transformer
- **规模**：~800M 参数（Chronos-Bolt）
- **Context Length**：2048 tokens

## 变体

| 变体 | 参数 | 特点 |
|------|------|------|
| Chronos-Tiny | 20M | 轻量级 |
| Chronos-Large | 200M | 高性能 |
| Chronos-Bolt | ~800M | 最新版本 |

## 与 UniCA 的结合

Chronos 是 UniCA 论文中验证的主要基础模型之一：

- **Chronos-Bolt + UniCA**：在单模态数据集上达到 0.506 MAPE（最优结果）
- **多模态任务**：Time-MMD 上达到 0.601 MAPE，提升 13%[^src-unca]

## 相关页面

- [[unified-covariate-adaptation]] — UniCA 框架
- [[timesfm]] — TimesFM 基础模型
- [[timesnet]] — TimesNet 基础模型
- [[most]] — MoST 多模态时空基础模型（不同领域：TS vs ST）
- [[aurora]] — Aurora 多模态生成式基础模型（Chronos 为单模态，Aurora 支持多模态）
- [[tats]] — TaTS 即插即用多模态框架（Chronos 为数值专用 tokenization，TaTS 通过辅助变量处理文本）

---

## 引用

[^src-chronos]: [[source-chronos]]
[^src-unca]: [[source-unca]]