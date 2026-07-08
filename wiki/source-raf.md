---
title: "RAF: Retrieval Augmented Time Series Forecasting"
type: source-summary
tags:
  - rag
  - retrieval-augmented
  - time-series
  - zero-shot
  - 2024
created: 2026-07-07
last_updated: 2026-07-07
source_count: 1
confidence: medium
status: active
---

# RAF: Retrieval Augmented Time Series Forecasting

**Authors**: Kutay Tire*, Ege Onur Taga*, M. Emrullah Ildiz, Samet Oymak (UT Austin / University of Michigan)

**Venue**: arXiv 2024 | **arXiv**: 2411.08249v2 | **Code**: [github.com/kutaytire/Retrieval-Augmented-Time-Series-Forecasting](https://github.com/kutaytire/Retrieval-Augmented-Time-Series-Forecasting)

## 核心贡献

RAF 是 **首个系统性地将检索增强生成（RAG）应用于时序基础模型（TSFM）零样本预测的框架**。[^src-raf] 核心发现是：现代 TSFM（如 Chronos）具有内在的检索能力——给定适当检索到的 motif（模体），它们能在零样本下对齐并复用它来改善预测，而无需微调。

## 关键设计

### 问题形式化

RAF 将时序检索定义为 **TS-R（Time-Series Retrieval）问题**：给定当前时间序列末尾的 motif m，在历史中找到其最匹配的同类 motif 及其后续序列，作为预测参考。[^src-raf]

论文理论证明（Theorem 1）：具有两层注意力机制和绝对位置编码的 Transformer 架构，通过步长=1 的 patch embedding，即可求解 TS-R 问题。

### RAF 框架

1. **数据库构建**：每个数据集预留 80% 序列构建检索数据库，严格防止信息泄露（仅包含严格早于查询窗口的时间步）。
2. **匹配与相似度度量**：使用 Chronos-Base 编码器提取嵌入，通过 ℓ2 范数（长上下文）或余弦相似度（短上下文）找到 top-1 最佳匹配。
3. **实例归一化**：对原始查询和检索序列分别归一化，消除分布偏移。
4. **检索查询拼接**：将检索到的序列（context + future）与原始 context 拼接，通过偏移对齐保证连续性，形成增强后的输入序列。
5. **零样本预测**：增强序列直接送入预训练 TSFM 进行预测，权重完全冻结。

### 两种模式

- **Naive RAF**：模型权重完全冻结，黑盒使用。
- **Advanced RAF**：额外对模型进行检索增强微调，进一步挖掘提升空间。

## 实验结果

### 模型规模效应

在 Benchmark I（C∈{50,75,100,150}, H∈{10,15,20}）和 Benchmark II 上，RAF 在 Chronos Mini 和 Base 上均超越基线，且 **大模型获益更显著**，与 LLM 领域 RAG 的规模定律一致。[^src-raf]

### 架构通用性

在 Chronos、Moirai、TimesFM、Lag-Llama 四种 TSFM 上验证，RAF 在所有架构上均有效。非 Transformer 基线（DLinear、LightGBM、ARIMA）的 RAF 增益很小或为负，说明检索增强需要注意力/跨序列融合能力。[^src-raf]

### 关键结论

- RAF 在跨领域零样本预测中收益最大（如交通→金融的领域迁移）。
- 检索增强是 **大型 TSFM 的涌现能力**：Chronos Mini 完全无法执行 TS-R，而 Small 和 Base 可以。
- Advanced RAF（带微调）全面超越所有对比方法。
- 检索对未来异常/突变事件的预测改善尤为显著。

## 相关链接

- [[source-time-llm|Time-LLM]] — 通过重编程冻结 LLM 进行时序预测，RAF 则利用 TSFM 的上下文学习
- [[chronos|Chronos]] — RAF 的主要评估对象，时序基础模型
- [[source-sundial|Sundial]] — 基于 Flow Matching 的时序基础模型系列
- [[retrieval-augmented-spatio-temporal-forecasting]] — 检索增强时空预测泛式

[^src-raf]: [[source-raf]]
