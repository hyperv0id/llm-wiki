---
title: "MiDDiR"
type: entity
tags:
  - time-series-forecasting
  - diffusion-models
  - probabilistic-forecasting
  - channel-dependency
  - retrieval-guidance
  - multivariate
  - iclr-2026
created: 2026-06-08
last_updated: 2026-06-09
source_count: 2
confidence: medium
status: active
---

# MiDDiR (Mixed Channel Dependency Diffusion Model with Retrieval Guidance)

**MiDDiR** 是一个面向多元时间序列概率预测的混合通道依赖扩散模型，由匿名作者提交至 ICLR 2026（双盲评审中），在 7 个真实世界数据集上取得 SOTA 概率预测性能[^src-middir]。

## 核心设计

### 混合通道依赖（Mixed Channel Dependency）

MiDDiR 的核心架构创新是通道依赖编码 + 通道独立去噪的非对称设计[^src-middir]：

| 阶段 | 策略 | 架构 | 设计理由 |
|------|------|------|---------|
| 编码 | Channel-Dependent (CD) | FC + Multi-head Attention | 捕获跨通道信息以获取丰富历史表示 |
| 去噪 | Channel-Independent (CI) | DiT block × N + AdaLN | 降低联合分布建模复杂度 |

详细原理见 [[mixed-channel-dependency]]。

### 检索引导（Retrieval Guidance）

推理时从训练集中检索相似历史模式，分析性偏置扩散得分估计[^src-middir]。这是**首个将检索引入扩散模型分析性引导的工作**。详细机制见 [[retrieval-guidance]]。

### 端到端训练

编码器（CD）和去噪网络（CI）联合训练，采用 ε-参数化损失最大化证据下界[^src-middir]。同时使用 [[instance-normalization|RevIN]] 减少非平稳性——训练阶段只对输入做归一化，输出阶段不对 ε-预测做逆归一化，仅在采样完成后逆归一化最终生成结果[^src-middir]。

## 性能

| 指标 | MiDDiR | NsDiff (次优) | TimeDiff | TMDM |
|------|--------|---------------|----------|------|
| CRPS (avg) | **0.243** | 0.311 | 0.518 | 0.343 |
| QICE (avg) | **2.322** | 4.344 | 14.649 | 3.933 |
| MAE (avg) | **0.336** (gen) | 0.401 | 0.540 | - |

GIFT-Eval：全部 39 个模型中 MSE/NMRSE 第 3，MAPE 非基础模型第 1（中/长区间）[^src-middir]。

## 消融发现

- **检索引导 λ**：中等 λ (~0.01–0.02) 最优；过高导致过拟合训练集（ETTm1），但高维数据集（Traffic）获益更大[^src-middir]
- **通道依赖编码**：移除 CD → 性能下降，多通道场景退化超 50%[^src-middir]
- **参数效率**：MiDDiR 参数量对通道数不敏感；NsDiff/TimeDiff/TMDM 随通道数快速增长[^src-middir]
- **检索开销极小**：单变量检索 0.054–0.176 ms，引导仅增加 0.51%–0.86% 采样步时间[^src-middir]

## 相关模型

- [[nsdiff|NsDiff]] — 被 MiDDiR 超越的 SOTA 扩散预测方法 (ICML 2025)
- [[timegrad|TimeGrad]] — 首个扩散时序预测模型 (ICML 2021)
- [[csdi|CSDI]] — 条件扩散时序插补/预测 (NeurIPS 2021)
- [[tedm|TEDM]] — EDM 框架下的扩散时序预测 (ICLR 2026)
- [[simdiff|SimDiff]] — 端到端扩散点预测 (AAAI 2026)
- [[gtr|GTR]] — 全局时序检索模块 (ICLR 2026)
- [[patchtst|PatchTST]] — CI 策略的开创者 (ICLR 2023)
- [[itransformer|iTransformer]] — attention 捕获跨变量相关 (ICLR 2024)
- [[ratd|RATD]] — 首个检索增强时序扩散模型（NeurIPS 2024），但以 RMA 注意力将检索结果作为条件特征输入，区别于 MiDDiR 的分析性得分倾斜[^src-ratd]

## 注意

- ⚠️ 论文处于 ICLR 2026 双盲评审中，尚未接收
- 代码未公开

[^src-middir]: [[source-middir]]
[^src-ratd]: [[source-ratd]]
