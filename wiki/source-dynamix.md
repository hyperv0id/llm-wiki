---
title: "DynaMix: True Zero-Shot Inference of Dynamical Systems Preserving Long-Term Statistics"
type: source-summary
tags:
  - dynamical-systems
  - zero-shot
  - foundation-model
  - mixture-of-experts
  - neurips2025
created: 2026-07-17
last_updated: 2026-07-17
source_count: 1
confidence: high
status: active
---

# Source: DynaMix

**Authors**: Christoph Jürgen Hemmer, Daniel Durstewitz — Heidelberg University / Central Institute of Mental Health, Mannheim. **Venue**: NeurIPS 2025.

## 核心论点

DynaMix 是首个能够从上下文信号零样本重建未知动力系统的 DSR 基础模型，无需任何重训练或微调[^src-dynamix]。其核心论点是：基于动力系统原理构建的基础模型可以比通用的时间序列基础模型（如 Chronos、TimesFM）更好地捕捉长期统计特性。

## 方法

模型采用混合专家（MoE）架构，使用 J=10 个 AL-RNN 专家（M=30 维潜在状态，其中 P=2 个 ReLU 单元），通过门控网络进行专家选择[^src-dynamix]。门控网络包含：状态注意力机制（计算投影潜在状态与上下文观测之间的距离）、CNN 上下文编码器、以及 MLP 生成专家权重。模型仅约 10k 参数。

训练使用稀疏教师强制（STF, τ=10），在 Gilpin (2022) 的 34 个不同 3D 动力系统上生成约 60 万条序列，配合 MSE 损失和方差正则化项[^src-dynamix]。

## 主要结果

在 54 个未见过的 3D 测试系统上，DynaMix 在 Dstsp（状态空间几何散度）和 DH（功率谱 Hellinger 距离）上显著且一致地超越所有 TS 基础模型（Chronos、TimesFM、Mamba4Cast、TTM、Panda），且在短期预测 MASE 上保持竞争力[^src-dynamix]。在真实世界数据（交通、天气、fMRI、EEG）上，DynaMix 甚至经常超越专门针对此类数据训练的 TS 基础模型，尽管其训练语料库完全由合成动力系统组成。

## 局限

主要聚焦于平稳时间序列；对尖峰、非平稳性、不恰当的嵌入可能导致失败。提供了非平稳数据的 Box-Cox + 趋势减法概念验证，以及通过延时嵌入或位置编码处理高维/一维数据的方法。

[^src-dynamix]: [[source-dynamix]]
