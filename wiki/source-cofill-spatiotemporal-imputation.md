---
title: "CoFILL: Spatiotemporal Data Imputation by Conditional Diffusion"
type: source-summary
tags:
  - diffusion-models
  - spatio-temporal
  - data-imputation
  - conditional-diffusion
created: 2026-05-11
last_updated: 2026-05-11
source_count: 1
confidence: high
status: active
---

# CoFILL 论文摘要

**CoFILL** (Conditional Diffusion Model based on Temporal-Frequency Spatiotemporal Imputation) 是一种用于时空数据填补的新型条件扩散框架。

## 核心贡献

1. **解决误差累积问题**：通过非递归的扩散框架结构，有效减少填补过程中的误差累积。

2. **双流特征处理架构**：同时处理时域和频域特征，通过 Cross-Attention 融合，捕捉快速变化和底层模式。

3. **在三个真实数据集上验证**：AQI-36（北京空气质量）、METR-LA（洛杉矶交通）、PEMS-BAY（旧金山湾区交通）。

## 核心方法

- **预处理**：forward interpolation + Gaussian noise ��种预填补策略
- **条件信息模块**：TCN（时域）+ GCN（空域）+ DCT（频域）→ Cross-Attention 融合
- **噪声预测网络**：Temporal Attention + Spatial Attention，条件信息引导去噪

## 性能

在 MAE、MSE、CRPS 三个指标上，CoFILL 在 15 种配置中的 12 种达到最优，相比 PriSTI 在 METR-LA Block 场景下 MAE/MSE 降低 10.22%。

## 代码

公开于 https://github.com/joyHJL/CoFILL

[^src-cofill]: [[source-cofill-spatiotemporal-imputation]]