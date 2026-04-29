---
title: "SimDiff"
type: entity
tags:
  - diffusion-models
  - time-series-forecasting
  - transformer
  - point-forecasting
  - aaaai-2026
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# SimDiff

SimDiff (Simpler Yet Better Diffusion Model) 是首个纯端到端的扩散模型，在时间序列点预测任务上取得 SOTA 结果，无需依赖任何外部预训练或联合训练的回归器 [^src-simdiff]。

## 核心创新

1. **统一 Transformer 架构**：单一 Transformer 网络同时作为去噪器和预测器 [^src-simdiff]
2. **Normalization Independence (N.I.)**：解耦训练时的归一化处理，缓解分布漂移 [^src-simdiff]
3. **Median-of-Means (MoM) 集成**：将扩散模型的概率样本聚合为精确点估计 [^src-simdiff]
4. **无跳跃连接设计**：避免时间序列中噪声放大问题 [^src-simdiff]

## 性能

- 9 个数据集平均 rank 1.33，超越 PatchTST (3.22) 和 mr-Diff (4.00) [^src-simdiff]
- MSE 相比其他扩散模型平均降低 8.3% [^src-simdiff]
- 推理速度比现有扩散方法提升超 90% [^src-simdiff]

## 相关模型

- 对比：[[autoformer]] (分解式 Transformer, NeurIPS 2021)
- 对比：[[fedformer]] (频域分解 Transformer, ICML 2022)
- 对比：[[timesnet]] (时序 2D 变化建模, ICLR 2023)
- 对比：[[tqn]] (Temporal Query Network, ICML 2025)
- 对比：[[sparsetsf]] (稀疏建模, TPAMI 2026)
- 基础：[[diffusion-model]] — 扩散模型理论基础
- 对比：[[cyclenet]] (周期残差学习, NeurIPS 2024)

## 相关技术

- 使用：[[normalization-independence]] — 扩散专属归一化技术
- 使用：[[median-of-means-ensemble]] — 概率到点估计的聚合方法
- 使用：[[patch-based-tokenization]] — 时间序列 patch 化
- 使用：[[channel-independence]] — 通道独立处理
- 相关：[[instance-normalization]] — RevIN 策略

[^src-simdiff]: [[source-simdiff]]