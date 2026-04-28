---
title: "Learnable Recurrent Cycles"
type: technique
tags:
  - time-series-forecasting
  - periodicity-modeling
  - neural-network
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# Learnable Recurrent Cycles

可学习循环周期是 CycleNet 中 RCF 技术的核心组件，用于显式建模时序数据的周期性模式[^src-cyclenet]。

## 定义

给定通道数 $D$ 和先验周期长度 $W$，可学习循环周期是一个可训练的参数矩阵 $Q \in \mathbb{R}^{W \times D}$[^src-cyclenet]。

- $W$：周期长度（如电力数据的日周期为 24，周周期为 168）[^src-cyclenet]
- $D$：通道数（变量数）[^src-cyclenet]

## 初始化与训练

- 初始化为零向量[^src-cyclenet]
- 与骨干网络一起通过梯度反向传播进行训练[^src-cyclenet]
- 训练后学习到的周期表示揭示了序列内部的循环模式[^src-cyclenet]

## 周期长度选择

周期长度 $W$ 取决于数据集的先验特征，应设置为数据集中最大的稳定周��[^src-cyclenet]：

| 数据集 | 周期长度 |
|--------|---------|
| ETTh1/2 | 24（日周期）|
| ETTm1/2 | 96（日周期）|
| Electricity | 168（日/周周期）|
| Solar-Energy | 144（日周期）|
| Traffic | 168（日/周周期）|
| Weather | 144（日周期）|

可通过自相关函数（ACF）进一步验证数据集的周期[^src-cyclenet]。

## 使用方式

通过循环复制操作，可将周期 $Q$ 扩展为任意长度的周期分量序列：

- 对历史输入 $x_{t-L+1:t}$：对齐并复制得到 $c_{t-L+1:t}$[^src-cyclenet]
- 对未来预测 $x_{t+1:t+H}$：对齐并复制得到 $c_{t+1:t+H}$[^src-cyclenet]

## 参数效率

RCF 只需额外 $W \times D$ 个可学习参数，无额外 MACs 计算量[^src-cyclenet]。

## 引用

[^src-cyclenet]: [[source-cyclenet]]