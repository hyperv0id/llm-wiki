---
title: "Temporal Query Network for Efficient Multivariate Time Series Forecasting"
type: source-summary
tags:
  - time-series
  - forecasting
  - multivariate
  - attention
  - periodicity
  - channel-dependence
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# Temporal Query Network (TQNet)

## 论文信息

- **标题**: Temporal Query Network for Efficient Multivariate Time Series Forecasting
- **作者**: Shengsheng Lin, Haojun Chen, Haijie Wu, Chunyun Qiu, Weiwei Lin (华南理工大学)
- **发表**: ICML 2025 (第 42 届国际机器学习会议)
- **代码**: https://github.com/ACAT-SCUT/TQNet

## 核心贡献

### Temporal Query (TQ) 技术

TQ 技术的核心思想是：用**周期性偏移的可学习向量**作为注意力机制中的 Query，从而在注意力中融合全局和局部相关性[^src-tqn]。

- **Query (Q)**：来自周期性偏移的可学习参数 θ_TQ ∈ R^{C×W}，捕捉变量间的全局相关性
- **Key (K) & Value (V)**：来自原始输入序列 X_t，编码样本级别的局部相关性
- **周期偏移机制**：每隔 W 个时间步，提取相同的 TQ 向量片段，实现参数复用和噪声平滑

### TQNet 架构

TQNet 仅由以下组件构成：
1. **单层 TQ-MHA** (Temporal Query-enhanced Multi-Head Attention)
2. **浅层 MLP** (两层 GeLU 激活)
3. **可选的 Instance Normalization**

这种极简架构在 12 个真实数据集上取得了 SOTA 性能，且计算效率与线性模型相当。

## 关键洞察

论文指出：真实世界数据中的非平稳干扰（极端值、缺失值、噪声）会导致**样本级相关性**与**全局相关性**产生显著差异。传统自注意力方法完全依赖样本级相关性，容易受到噪声干扰。TQ 技术通过将全局先验（可学习向量）注入注意力，使模型能够建立更稳定的变量间依赖关系。

## 实验结果

- 在 12 个数据集上取得 Top 2 性能（24/24 个预测指标）
- 在高维多变量数据集（Electricity, PEMS）上优势明显
- 推理效率与 DLinear 等线性模型相当

## 与本 Wiki 已收录论文的关系

- **与 CycleNet 的关联**：TQ 技术受 CycleNet 启发，采用可学习参数捕捉周期模式[^src-tqn]
- **与 iTransformer 的对比**：iTransformer 完全依赖样本级自注意力，而 TQNet 通过 TQ 技术融合全局先验
- **与 PENGUIN 的对比**：两者都将周期先验注入注意力，但 TQ 使用可学习向量作为 Query，PENGUIN 使用周期分组注意力偏置

## 局限性

- 需要预先知道周期长度 W（通过 ACF 或先验知识确定）
- 在非平稳数据集上性能弱于某些方法

[^src-tqn]: [[source-tqn]]