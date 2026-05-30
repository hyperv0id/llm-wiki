---
title: "MPGRU"
type: technique
tags:
  - graph-neural-network
  - recurrent-network
  - message-passing
  - spatio-temporal
  - imputation
created: 2026-05-30
last_updated: 2026-05-30
source_count: 1
confidence: medium
status: active
---

# MPGRU

**MPGRU** (Message-Passing GRU) 是 [[grin]] 的核心编码单元，将标准 GRU 的门控操作替换为消息传递层（MPNN），在更新隐藏状态的同时聚合邻居节点的空间信息[^src-2108-00298]。

## 公式

给定节点 $i$ 在时间步 $t$ 的输入，MPGRU 的三个门控/候选状态为：

$$r_t^i = \sigma\left(\text{MPNN}\left(\hat{x}_t^{i(2)} \| m_t^i \| h_{t-1}^i, W_t\right)\right)$$

$$u_t^i = \sigma\left(\text{MPNN}\left(\hat{x}_t^{i(2)} \| m_t^i \| h_{t-1}^i, W_t\right)\right)$$

$$c_t^i = \tanh\left(\text{MPNN}\left(\hat{x}_t^{i(2)} \| m_t^i \| r_t^i \odot h_{t-1}^i, W_t\right)\right)$$

$$h_t^i = u_t^i \odot h_{t-1}^i + (1 - u_t^i) \odot c_t^i$$

其中 $\hat{x}_t^{i(2)}$ 是上一时间步解码器的输出（第二阶段填补值），$m_t^i$ 是缺失掩码，$h_{t-1}^i$ 是前一时刻的隐藏状态，$W_t$ 是图邻接矩阵[^src-2108-00298]。

## 设计要点

- **门控即消息传递**：每个门控操作不仅是线性变换+sigmoid，而是一个完整的 MPNN 前向传播，包含邻居聚合
- **扩散卷积**：GRIN 使用扩散卷积（diffusion convolution, $k=2$）作为 MPNN 算子，建模双向图扩散过程
- **缺失感知输入**：将缺失掩码 $m_t^i$ 拼接到输入中，让模型区分观测值和填补值
- **填补值反馈**：使用上一时间步解码器的输出（而非原始缺失值）作为下一步输入，使编码器能利用自身填补结果

## 与 DCRNN 的关系

MPGRU 的结构与 DCRNN (Li et al., 2018) 类似——都使用图卷积替代 GRU 的门控线性变换。关键区别：DCRNN 用于预测任务，MPGRU 专用于填补任务，配合空间解码器形成两阶段填补流程。

## 参数效率

GRIN 使用 64 个隐藏单元（编码器和解码器各 64），总参数约 200K，远小于 BRITS 的 ~4M 参数量，在参数效率上显著更优[^src-2108-00298]。

[^src-2108-00298]: [[source-2108-00298]]
