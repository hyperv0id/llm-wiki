---
title: "CycleNet"
type: entity
tags:
  - time-series-forecasting
  - periodicity-modeling
  - neurips-2024
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# CycleNet

CycleNet 是华南理工大学等机构提出的长期时序预测模型（NeurIPS 2024），通过显式建模周期性模式实现高效准确的预测[^src-cyclenet]。

## 模型架构

CycleNet 由两部分组成：

1. **Residual Cycle Forecasting (RCF)**：核心创新技术，使用可学习的循环周期 $Q \in \mathbb{R}^{W \times D}$ 建模数据内在周期[^src-cyclenet]
2. **骨干网络**：单层 Linear（CycleNet/Linear）或双层 MLP（CycleNet/MLP）[^src-cyclenet]

### RCF 技术流程

1. **周期建模**：生成可学习循环周期 $Q$，通过循环复制得到周期分量 $C$[^src-cyclenet]
2. **残差预测**：从原始输入中移除周期分量，对残差进行预测，最后加回周期分量[^src-cyclenet]

## 性能表现

| 数据集 | CycleNet/MLP 排名 |
|--------|-------------------|
| ETTh1/2 | 第一 |
| ETTm1/2 | 第一 |
| Electricity | 第一 |
| Solar-Energy | 第一 |
| Weather | 第一 |
| Traffic | 第二（仅次于 iTransformer）|

## 效率优势

相比 iTransformer，CycleNet/MLP 参数减少超过 10 倍（472.9K vs 5.15M），MACs 减少超过 12 倍[^src-cyclenet]。

## 即插即用

RCF 技术可作为即插即用模块，显著提升现有模型（如 PatchTST、iTransformer）的预测精度[^src-cyclenet]。

## 与其他模型的关系

- **与 DLinear**：CycleNet/Linear 与 DLinear 使用相同的单层线性骨干，但 RCF 技术带来显著性能提升[^src-cyclenet]
- **与 Autoformer/FEDformer**：RCF 本质上是一种季节性-趋势分解方法，但显式建模全局周期模式[^src-cyclenet]
- **与 SparseTSF**：两者都利用周期信息，但 RCF 使用可学习周期而非跨周期稀疏预测[^src-cyclenet]

## 局限性

- 在具有显著时空特性和极端值的 Traffic 数据集上性能略逊于 iTransformer[^src-cyclenet]
- 当前版本仅考虑单通道关系建模[^src-cyclenet]

## 引用

[^src-cyclenet]: [[source-cyclenet]]