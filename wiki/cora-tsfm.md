---
title: "CoRA: Covariate-Aware Adaptation of TSFMs"
type: entity
tags:
  - time-series-foundation-model
  - covariate-adaptation
  - granger-causality
  - multimodal
  - iclr-2026
last_updated: 2026-07-28
source_count: 6
confidence: high
status: active
---

# CoRA: Covariate-Aware Adaptation

> [!warning] 同缩写消歧
> 本页是 **Covariate-awaRe Adaptation**（外生协变量注入）。另有一篇 ICLR 2026 的 **CoRrelation-aware Adapter**（通道相关插件），见 [[cora-correlation-aware-adapter]] / [[source-cheng-2025-cora-correlation-aware-adapter]]。

**CoRA** (Covariate-awaRe Adaptation) 是一个面向时间序列基础模型（TSFMs）的通用、可解释、零初始化协变量适配框架，ICLR 2026 投稿[^src-cora]。

## 动机

大多数 TSFMs（[[timesfm|TimesFM]]、[[chronos|Chronos]]、[[sundial|Sundial]]、Moirai）在单变量时间序列上预训练，天然忽略了对真实预测任务至关重要的外生协变量——包括时间序列、文本、图像等多模态信息[^src-cora]。现有适配方法（[[chronosx|ChronosX]]、AdaPTS、[[unica|UniCA]]）虽可注入协变量，但 ChronosX 的 IIB 在编码器前改 token 嵌入、且整体**无零初始化**，易扰动预训练空间并导致不稳定/遗忘；UniCA 的 CAP 前置融合同属此类风险[^src-cora][^src-unica][^src-chronosx]。

> [!note] ChronosX 机制补全
> [[chronosx|ChronosX]]（AISTATS 2025）= **IIB（past→嵌入）+ OIB（future→logits）** 模块适配，可冻结 backbone；另有 TimesFMX/MOMENTX 与 32 合成协变量基准。CoRA 表中 ChronosX avg MSE 0.134 为 **Sundial 公平对比协议下的外部对照**，与 ChronosX 原文 18 集 WQL 设定不同[^src-chronosx][^src-cora]。

CoRA 被设计来解决一个更根本的问题：**如何在保持预训练模型完整性的前提下，渐进式地融合异构协变量信息。**

## 三大设计原则

### 原则 1：基础模型作为冻结特征提取器

CoRA 不修改任何预训练模型的主干网络。对每种协变量模态，使用对应的冻结模型提取特征：

- **时间序列协变量** → 冻结的 TSFM backbone（取最后一步嵌入）
- **文本协变量** → 冻结的 LLM backbone（Qwen3-Embedding），时间步平均池化
- **图像协变量** → 冻结的视觉 backbone（ViT），时间步平均池化

冻结嵌入在实验中证明比原始数据更具信息量[^src-cora]。

### 原则 2：因果嵌入（Causality Embedding）实现可解释选择

多协变量预测的核心难点在于：并非所有协变量都对目标变量有真正的预测价值。CoRA 引入可训练的 `W_CE ∈ ℝ^N`（N = 协变量总数），经 Softmax 归一化后对协变量嵌入加权求和。

架构流程：
```
多模态协变量 → 冻结特征提取 → 线性投影到统一空间 → Causality Embedding (Softmax加权) → 统一嵌入 H
```

实验表明，学习到的因果权重与传统 Granger-Geweke 统计因果检验高度相关（Pearson 相关系数图证实），使 CoRA 的选择机制具有明确的统计可解释性[^src-cora]。因果嵌入与简单相关的区别在于：Granger 因果捕获的是协变量对目标变量的预测有用性，而非直接因果关系——例如，协变量可能与目标变量零相关但仍具有显著的 Granger 因果性（如 sine/cosine 对）[^src-cora]。

### 原则 3：零初始化条件注入，渐进式融合

CoRA 通过 adaLN（adaptive layer normalization，源自 [[dit|DiT]]）将统一协变量嵌入注入 TSFM 的预测头：

```
H → MLP → γ (shift), β (scale), α (output scaling)
预测头前：γ ⊙ head_input + β
预测头后：(1+α) × head_output
```

所有新增参数（投影矩阵、MLP）均零初始化，保证适配起点与原始 TSFM 完全等价。这意味着模型从预训练的零样本能力出发，逐渐融合协变量信息，避免了灾难性遗忘[^src-cora]。消融实验证明：替换零初始化为 Xavier 初始化导致性能退化 4.3%[^src-cora]。该设计遵循 [[zero-initialized-adaptation|零初始化适配]] 原则，与 LoRA 的 B 矩阵零初始化和 DiT 的 adaLN-Zero 一脉相承[^src-dit]。

## 协变量处理的三条路径

CoRA 通过 `[[channel-independence|Channel Independence]]` 机制处理多元目标变量：每个目标变量独立送入 TSFM，共享相同的协变量适配权重。这种设计与 TSFM 的单变量预训练兼容。

根据协变量的时间特性：
- **未来未知协变量**（τ = T）：仅历史可用
- **未来已知协变量**（τ = T+H）：历史+未来均可用
- **静态协变量**（τ = 1）：单个快照

## 性能总结

### 单模态协变量预测（TSLib 基准）

在以 [[sundial|Sundial]] 为 backbone 的对比中，CoRA 在 ETTh1/ETTh2/ETTm1/ETTm2/Weather/ECL/Traffic 七个数据集上全面超越所有适配方法和监督深度模型：

| 对比方法 | avg MSE | CoRA 相对降低 |
|----------|---------|---------------|
| AdaPTS | 0.084 | 19.0% |
| ChronosX | 0.134 | 49.3% |
| UniCA | 0.084 | 19.0% |
| TimeXer | 0.090 | 24.4% |

*Avg across 7 datasets for prediction horizons {96, 192, 336, 720}*

### 多模态预测

- **RT-1**（图像协变量）：超越最佳端到端监督模型 12.7% MSE[^src-cora]
- **[[time-mmd|Time-MMD]]**（文本协变量）：超越 UniCA 1.9% MSE[^src-cora][^src-time-mmd]

### Few-Shot 预测（EPF）

在数据稀缺场景下（1%~25% 训练数据），CoRA 的优势更加显著：端到端模型 TimeXer 在此条件下严重退化，而 CoRA 仅需极少量样本即可维持高性能[^src-cora]。

### 多元预测

CoRA 在 7 个多元数据集上平均 MSE 降低 14.5%，超越 TimeXer。优势来源：预训练 TSFM 已内化通用时间模式，CoRA 在此基础上精准捕获跨变量依赖[^src-cora]。CoRA 与 [[dits|DiTS]] 代表了协变量感知预测的两种不同路线：前者冻结现有 TSFM 后置注入，后者从头构建 MM-DiT 双流架构[^src-dits]。

### 跨 Backbone 泛化

| Backbone | MSE 降低 |
|----------|----------|
| Sundial | 14.2% |
| TimesFM | 3.3% |
| Chronos-Bolt | 4.9% |
| FlowState | 3.3% |

## 与 UniCA 的结构性对比

| 维度 | CoRA | [[unica|UniCA]] |
|------|------|------|
| 协变量注入位置 | 预测头（后置注入） | 编码器前后（前置+后置融合） |
| 预训练空间保护 | 严格保护（冻结 backbone + 零初始化） | 前置注入扰乱了编码器输入空间 |
| 协变量选择 | Causality Embedding（可解释） | 注意力池化（黑盒） |
| 零初始化 | 是 | 否 |
| 性能（[[time-mmd|Time-MMD]]） | MSE 0.641 | MSE 0.653 |

## 局限性

1. 未讨论协变量数量 N 极大时的可扩展性（Causality Embedding 的维度为 N）
2. adaLN 假设所有协变量条件信号具有同质性，未纳入模态特有的结构性先验
3. 存在 self-referential 隐忧：因果嵌入的预测质量依赖抽样分布的稳定性，但条件注入本身改变了分布——这一预设未被讨论[^src-cora]

## 相关页面

- [[timesfm]] — 核心兼容 TSFM backbone 之一
- [[chronos]] — 核心兼容 TSFM backbone 之一
- [[sundial]] — 实验主要 backbone，CoRA 在其上取得最佳适配效果
- [[dits]] — 协变量感知预测的替代路线（MM-DiT 双流架构）
- [[tsfm-covariate-adaptation-comparison]] — 六种 TSFM 适配方法的系统对比（含 ChronosX 机制锚点）
- [[chronosx]] — 早期模块适配基线（IIB+OIB）
- [[source-chronosx]] — ChronosX 源摘要
- [[zero-initialized-adaptation]] — CoRA 零初始化设计的理论基础
- [[channel-independence]] — CoRA 在多元预测中采用的策略
- [[cross-dimension-dependency]] — CoRA 通过 CI 隐式处理的跨变量依赖概念
- [[heterogeneous-covariates]] — 异构协变量的分类与处理挑战
- [[multimodal-time-series-forecasting]] — 多模态预测的总体概念
- [[time-mmd]] — 文本协变量多领域基准（NeurIPS 2024 D&B）
- [[source-time-mmd]] — Time-MMD 源摘要
- [[unified-covariate-adaptation]] — UniCA 的详细介绍（对比方法）
- [[covariate-homogenization]] — UniCA 的协变量同质化技术（对比）
- [[conditional-attention-pooling]] — UniCA 的融合机制（对比）

## 引用

[^src-cora]: [[source-cora]]
[^src-unica]: [[source-unica]]
[^src-dit]: [[source-dit]]
[^src-dits]: [[source-dits]]
[^src-sundial]: [[source-sundial]]
[^src-chronosx]: [[source-chronosx]]
[^src-time-mmd]: [[source-time-mmd]]
