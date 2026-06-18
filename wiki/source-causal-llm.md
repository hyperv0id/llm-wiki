---
title: "Causal-LLM: Towards Predictive and Interpretable Spatiotemporal Foundation Models"
type: source-summary
tags:
  - spatio-temporal-forecasting
  - foundation-model
  - causal-reasoning
  - large-language-models
  - interpretability
  - neuro-symbolic
created: 2026-06-18
last_updated: 2026-06-18
source_count: 1
confidence: medium
status: active
---

# Causal-LLM: Towards Predictive and Interpretable Spatiotemporal Foundation Models

这是一篇 AAAI-26 的愿景/提案论文（proposal paper），提出 Causal-LLM——一个神经符号（neuro-symbolic）时空基础模型，旨在同时实现预测能力和因果可解释性。[^src-causal-llm]

---

## 核心问题

STGNN 在预测精度上取得了显著进展，但存在根本性的不透明性：它们能回答"预测值是多少"（what），却无法解释"为什么是这个预测值"（why）。在气象、城市规划和公共卫生等高 stakes 领域，这种缺乏可解释性构成了信任瓶颈。[^src-causal-llm]

**核心论点**：真正的可解释性不能是事后附加的（afterthought），必须内建于模型的核心学习过程中。

---

## 方法论

### 模型架构：三阶段管道

基于 TimeLLM 架构的改编：[^src-causal-llm]

1. **时空编码器（GNN）**：处理多变量时空数据，学习系统状态的压缩潜在向量。GNN 天然适合传感器网络的图结构拓扑。
2. **重编程模块（Reprogramming Module）**：神经符号桥接层，将 GNN 的连续潜在向量投影到 LLM 的词嵌入空间，生成"物理状态令牌"（physical state tokens）。每个令牌代表一个可学习的、重复出现的物理现象。冻结的预训练 LLM 无需修改权重即可处理这些复杂动态。
3. **多任务解码器**：LLM 的最终输出分别导向两个头——通过线性头生成数值预测，通过语言头生成自然语言因果解释。

### 训练方法：因果数据合成（Causal Data Synthesis）

这是该论文的核心创新。不同于让模型从原始数据中隐式发现相关性，作者提出通过显式引导训练：

- 为每个重大历史事件构建**三元组训练数据**：事件前的原始数据（输入）→ 事件数值的真值（预测目标）→ 人类编写的因果解释（解释目标），例如"高压系统形成温度逆温层，将污染物困住"。
- 通过多任务联合损失端到端训练，模型被显式教导：GNN 感知到的特定物理状态不仅对应数值结果，还对应特定的因果叙事。

### 评估维度

- **预测精度**：标准指标（MAE、RMSE），在领域数据集和通用基准（ETT）上评估
- **解释质量**：定量 NLP 指标（ROUGE、BERTScore）+ 人类领域专家定性评估（事实正确性、连贯性、科学实用性）

---

## 相关工作定位

作者将现有 LLM+时间序列方法分为两类：[^src-causal-llm]

1. **对齐方法（Aligning-based）**：将时间序列嵌入投影到 LLM 表示空间，创建共享潜在空间。但融合后的潜在空间仍是"黑盒"，且不完全适合图结构时空数据。
2. **提示方法（Prompting-based）**：将数值数据格式化为自然语言提示，利用 LLM 的上下文学习能力进行零样本预测。但缺乏物理基础，LLM 没有对科学时空数据背后物理定律的内在理解。

Causal-LLM 旨在超越这两种范式，实现"可解释性设计"。

---

## 作者背景

作者 Zhiqing Cui 的背景与此项目直接相关：[^src-causal-llm]

- **时空预测**：CauAir（全国空气质量预测的因果数据集）、Prithvi-TC（热带气旋预测基础模型）——这些模型虽达到 SOTA 精度，但无法提供人类可理解的解释。
- **多模态推理**：Draw with Thought（DWT）项目——将非结构化图表图像转换为结构化符号 XML。

---

## 贡献与局限

### 贡献
1. 提出神经符号基础模型的新范式：GNN 感知 + LLM 推理
2. 因果数据合成训练方法，将可解释性作为一阶设计目标
3. 为科学领域的可信 AI 提供蓝图

### 局限
- 这是一篇提案/愿景论文，尚未完成实验验证
- 因果数据合成需要大量人工标注的因果解释，数据获取成本高
- 可能存在数值精度与解释质量之间的权衡
- 图结构数据到 LLM 令牌空间的映射可能丢失细粒度空间信息

## 引用

[^src-causal-llm]: [[source-causal-llm]]