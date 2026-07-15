---
title: "Dynamical Systems Reconstruction"
type: concept
tags:
  - dynamical-systems
  - time-series
  - generative-modeling
created: 2026-07-17
last_updated: 2026-07-17
source_count: 1
confidence: medium
status: active
---

# Dynamical Systems Reconstruction (DSR)

**动力系统重建（DSR）** 是指从观测时间序列数据中学习生成式替代模型，使其能够复现底层动力系统的长期行为——包括状态空间（吸引子）的拓扑、几何特性，以及时间域中的不变统计量或"气候"统计[^src-dynamix]。

## 与时间序列预测的区别

DSR 超越了传统时间序列预测[^src-dynamix]：

| 维度 | 时间序列预测 | DSR |
|------|-------------|-----|
| 目标 | 准确短期预测 | 复现长期统计行为 |
| 评估 | MAE/MSE（短视界） | Dstsp, DH, Lyapunov 指数（长视界） |
| 本质 | 条件分布拟合 | 生成式动力学替代模型 |

在混沌系统中，邻近轨迹呈指数分离，因此长期点预测在原则上不可行。DSR 转而关注系统的不变特性——吸引子几何和功率谱[^src-dynamix]。

## 主要方法

### 基于 RNN

- PLRNN / AL-RNN：分段线性/几乎线性的 RNN，配合稀疏教师强制训练
- 储备池计算（Reservoir Computing）：高维固定随机动力学 + 线性读出

### 基于微分方程

- Neural ODE：用神经网络参数化向量场 $\dot{x} = f_\theta(x)$
- 连续时间专家模型

### 基于算子

- Koopman 算子：通过提升到无穷维观测空间实现动力学线性化
- SINDy：从预定义函数库稀疏识别控制方程

## 关键训练技术

- **稀疏教师强制（STF）**：按固定间隔以数据推断状态替换前向迭代状态，控制梯度爆炸同时允许模型"探索未来"[^src-dynamix]
- **广义教师强制**：STF 的推广形式
- **长期统计正则化**：在损失中加入 Lyapunov 谱、分形几何或不变测度约束

## 评估指标

- **Dstsp**（KL 散度）：真实轨迹与生成轨迹在状态空间中的分布重叠度
- **DH**（Hellinger 距离）：功率谱之间的一致程度
- **最大 Lyapunov 指数**：轨迹指数分离速率，区分混沌（>0）与周期（≈0）行为

## 外域泛化

传统 DSR 方法需要针对每个具体系统进行定制训练，缺乏零样本和上下文推理能力[^src-dynamix]。[[dynamix|DynaMix]] 是首个实现真正零样本 DSR 的基础模型，标志着该领域的范式转变。

[^src-dynamix]: [[source-dynamix]]
