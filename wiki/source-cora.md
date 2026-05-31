---
title: "CoRA: Covariate-Aware Adaptation of Time Series Foundation Models"
type: source-summary
tags:
  - time-series-foundation-model
  - covariate-adaptation
  - multimodal-time-series
  - granger-causality
  - iclr-2026
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: medium
status: active
---

# CoRA 源文件摘要

**来源**: Anonymous authors (double-blind review). *CoRA: Covariate-Aware Adaptation of Time Series Foundation Models.* Under review at ICLR 2026.

## 核心论点

### 1. 问题：TSFM 在协变量感知预测上的空白

大多数时间序列基础模型（TSFMs）在单变量时间序列上预训练，无法利用现实预测任务中的多样协变量信息——包括分类变量、图像、文本等异构模态[^src-cora]。现有的协变量适配方法（ChronosX、AdaPTS、UniCA）虽然在 TSFM 编码器前注入协变量信号，但这种设计会扰乱预训练的嵌入空间，且缺乏零初始化，容易导致灾难性遗忘[^src-cora]。

### 2. 核心贡献：CoRA 三大组件

**CoRA** (Covariate-awaRe Adaptation) 提出了三个关键设计来优雅地解决上述问题：

#### (1) 冻结基础模型作为特征提取器
CoRA 冻结所有预训练基础模型（TSFM、LLM、视觉模型），仅提取其最后一层前的嵌入。实验证明，这些冻结嵌入比原始数据更具信息量[^src-cora]。对于时间序列协变量，取最后一步的嵌入；对于文本和图像，取所有时间步的平均嵌入。

#### (2) 因果嵌入（Causality Embedding）
CoRA 引入可训练的 `W_CE ∈ ℝ^N` 向量，自动学习每个协变量对目标变量的 Granger 因果显著度。该向量经 Softmax 归一化后对协变量嵌入加权求和，实现可解释的协变量选择。实验证明，学习到的因果嵌入与传统 Granger-Geweke 因果检验高度一致[^src-cora]。

#### (3) 零初始化条件注入
CoRA 通过 adaLN (adaptive layer normalization) 将加权协变量嵌入注入 TSFM 预测头：生成 scale (γ) 和 shift (β) 参数调制预测头前后的统计量，外加 α 缩放预测结果。所有新增参数（投影矩阵 W_mi、b_mi、MLP）均零初始化，确保适配起点与预训练模型完全等价，渐进式融合外部信息而避免灾难性遗忘[^src-cora]。

### 3. 即插即用兼容性

CoRA 已验证与多种 TSFM 兼容：Sundial、TimesFM、Chronos-Bolt、FlowState、Moirai[^src-cora]。在全部 backbone 上均取得一致的性能提升（MSE 降低 3.3%~14.2%），证明其架构无关。

## 主要实验发现

| 任务 | 数据集 | CoRA 性能 |
|------|--------|-----------|
| 单模态协变量预测 | ETTh/ETTm/Weather/ECL/Traffic | 比次优方法 TimeXer 降低 ~14.5% MSE |
| 短时协变量预测 | EPF（5个电力市场） | 比 TimeXer 和 ChronosX 优势显著，数据稀缺下更突出 |
| 多模态预测 | RT-1（图像协变量） | 比最佳监督模型降低 12.7% MSE |
| 多模态预测 | Time-MMD（文本协变量） | 比 UniCA 降低 1.9% MSE |
| 多元预测 | TSLib 7 数据集 | 比 TimeXer 降低 14.5% MSE |

消融实验揭示：(1) 无协变量→性能退化 6.5% MSE；(2) 无 adaLN（直接加条件到输入）→退化 12.9%；(3) 无因果选择（均值聚合）→退化 8.3%；(4) 无零初始化→退化 4.3%[^src-cora]。

## 与 UniCA 的关键差异

UniCA 将协变量在 TSFM 编码器前注入（通过 CAP + 自注意力），而 CoRA 将协变量作为外部条件注入预测头。CoRA 的三个增量优势：(1) 严格保持预训练嵌入空间不变；(2) 因果嵌入实现可解释的协变量选择；(3) 零初始化保证渐进适配[^src-cora]。此外，CoRA 在 Time-MMD 和 RT-1 上均优于 UniCA。

## 局限性（隐性）

1. 未讨论协变量数量极大（N → ∞）时的可扩展性
2. adaLN 的 scale/shift 机制假设了条件信号的同质性，未考虑不同协变量的模态间结构性先验
3. 假设采样分布在条件注入 nudge 后保持稳定（self-referential 隐忧）——因果嵌入的预测质量依赖于分布稳定性，但注入本身改变了分布

## 与现有技术的关系

| 方法 | 与 CoRA 的关系 |
|------|-----------------|
| ChronosX | 前置注入协变量，无零初始化，CoRA 优于它 |
| AdaPTS | 前置注入，CoRA 将协变量保留在预测头 |
| UniCA | 同为 ICLR 2026，CAP 前置融合 vs CoRA 零初始化后置注入 |
| LoRA | 低秩适配，CoRA 借鉴了其零初始化思想 |
| DiT (Peebles & Xie) | CoRA 的 adaLN 机制直接来源于此 |

## 引用

[^src-cora]: [[cora-tsfm]]
