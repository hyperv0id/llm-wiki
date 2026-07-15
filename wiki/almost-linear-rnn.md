---
title: "Almost-Linear RNN"
type: technique
tags:
  - rnn
  - dynamical-systems
  - architecture
created: 2026-07-17
last_updated: 2026-07-17
source_count: 1
confidence: medium
status: active
---

# Almost-Linear RNN (AL-RNN)

**AL-RNN** 是 Brenner et al. (NeurIPS 2024) 提出的 RNN 变体，专为动力系统重建（DSR）设计，具有拓扑简约性和可解释性[^src-dynamix]。

## 定义

AL-RNN 描述 M 维潜在过程 $z_t \in \mathbb{R}^M$ 的演化：

$$z_t = A z_{t-1} + W \Phi^*(z_{t-1}) + h$$

其中：
- $A \in \text{diag}(\mathbb{R}^M)$：线性自连接（对角矩阵）
- $W \in \mathbb{R}^{M \times M}$：权重矩阵
- $h \in \mathbb{R}^M$：偏置项
- $\Phi^*$：仅在最后 P 个单元上施加 ReLU，前 M−P 个保持线性

$$\Phi^*(z_t) := [z_{1,t}, \cdots, z_{M-P,t}, \max(0, z_{M-P+1,t}), \cdots, \max(0, z_{M,t})]^T$$

前 N 个单元作为读出层，提供预测观测 $\hat{x}_t = z_{1:N,t}$[^src-dynamix]。

## 关键性质

- **近乎线性**：P << M（如 P=2, M=30），大部分动力学保持线性
- **拓扑简约**：少量非线性单元迫使模型以最简洁的方式编码动力学结构
- **可解释**：线性部分可进行谱分析，非线性部分可精确定位
- **参数高效**：极低参数量（DynaMix 中 10 个 AL-RNN 专家总计约 10k 参数），即可实现 SOTA DSR

## 在 DynaMix 中的角色

[[dynamix|DynaMix]] 使用 J=10 个 AL-RNN 作为混合专家的基础单元。消融实验证明，将 AL-RNN 替换为 LSTM、普通 RNN 或储备池计算均导致性能下降，这与 AL-RNN 配合 STF 训练的 SOTA DSR 能力一致[^src-dynamix]。

## 初始化

遵循 Brenner et al. 的协议：W 从高斯 $\mathcal{N}(0, 0.011^2)$ 抽取，h=0，A 设为归一化正定随机矩阵的对角线。初始潜在状态从上下文信号估计，包含一个联合学习的线性映射 L[^src-dynamix]。

[^src-dynamix]: [[source-dynamix]]
