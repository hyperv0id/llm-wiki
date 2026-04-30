---
title: "UniCA"
type: entity
tags:
  - time-series-foundation-model
  - covariate-adaptation
  - multimodal-time-series
  - iclr-2026
created: 2026-04-29
last_updated: 2026-04-29
source_count: 1
confidence: high
status: active
---

# UniCA (Unified Covariate Adaptation)

**UniCA** 是南京大学与蚂蚁集团联合提出的时间序列基础模型（Time Series Foundation Model, TSFM）协变量适应框架，发表于 ICLR 2026。论文由陆涵、刘宇、李岚等作者共同完成，南京大学的叶瀚嘉、湛德川担任通讯作者[^source]。

## 背景与问题

时间序列基础模型（如 Chronos、TimesFM、Time-MoE、Moirai）在预训练阶段仅使用实值数值序列。然而，现实世界的时序任务往往伴随丰富的**异构协变量**：

- **分类变量**：星期几、天气类型、节假日
- **图像**：雷达回波图、卫星云图
- **文本**：新闻标题、财经报告

这些协变量在预训练时不可用，却在推理时作为任务特定信息出现，导致 TSFM 无法有效利用它们。这种现象被称为 **模态鸿沟（modality gap）**[^src-unica]。

## 技术架构

UniCA 包含两个核心模块来解决异构协变量适应问题：

### 1. 协变量同质化（Covariate Homogenization, CH）

将不同模态的协变量转换为统一的高层次序列表示。通过：
- **分类变量**：嵌入表编码
- **图像**：CLIP 编码器 + 线性投影
- **文本**：BERT 编码器 + 线性投影

投影后的表示可与实值时间序列直接一起处理[^src-unica]。

### 2. 注意力双融合模块（Attention-based Dual Fusion）

- **预融合模块**：在 TSFM 编码器之前，使用条件注意力池化聚合协变量信息
- **后融合模块**：在 TSFM 编码器之后，通过自注意力融合已编码的历史和未来协变量

关键设计：**保持 TSFM 主干冻结**，仅训练轻量级融合模块，保留预训练模型的泛化能力[^src-unica]。

## 主要特性

| 特性 | 描述 |
|------|------|
| 架构无关 | 已在 Chronos-Bolt、TimesFM-2、Time-MoE、Moirai 上验证 |
| 即插即用 | 无需修改 TSFM 主体，仅添加轻量模块 |
| 轻量级 | 额外计算开销可忽略 |
| 多模态支持 | 支持分类、图像、文本协变量 |
| SOTA 性能 | 在 12 个单模态和 2 个多模态基准上超越专门模型和现有适应方法 |

## 实验验证

UniCA 在以下基准上验证有效性：
- **单模态**（12 个数据集）：能源、交通、医疗等领域的时序预测
- **多模态**：MMSP（图像）、Time-MMD（文本）

相比 ChronosX、TTM-R2、线性回归等基线方法，UniCA 在 MAE、MAPE、MSE、CRPS 等指标上取得一致提升[^src-unica]。

## 局限性

1. 假设协变量与目标序列时间对齐
2. 对噪声协变量敏感
3. 无不确定性感知融合
4. 不支持非对齐或部分观测协变量

## 相关页面

- [[source-unica]] — 源文件摘要
- [[unified-covariate-adaptation]] — 统一协变量适应概念
- [[covariate-homogenization]] — 协变量同质化技术
- [[covariate-fusion-module]] — 协变量融合模块技术
- [[timesnet|TimesNet]] — 另一种时间序列基础模型
- [[instance-normalization|RevIN]] — 协变量适应的传统方法（分布漂移）
- [[normalization-independence]] — SimDiff 的归一化独立技术
- [[tslib|TSLib]] — 时间序列模型基准库

## 参考文献

[^src-unica]: [[source-unica]]
[^source]: https://arxiv.org/abs/2506.22039 — UniCA: Unified Covariate Adaptation for Time Series Foundation Model (ICLR 2026)