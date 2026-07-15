---
title: "DynaMix"
type: entity
tags:
  - dynamical-systems
  - foundation-model
  - mixture-of-experts
  - zero-shot
  - neurips2025
created: 2026-07-17
last_updated: 2026-07-17
source_count: 1
confidence: medium
status: active
---

# DynaMix

**DynaMix** 是首个面向[[dynamical-systems-reconstruction|动力系统重建（DSR）]]的零样本基础模型，由 Hemmer & Durstewitz (NeurIPS 2025) 提出[^src-dynamix]。它从一段上下文信号出发，无需重训练即可忠实地预测新动力系统的长期演化，包括吸引子几何和功率谱等不变统计量。

## 架构

DynaMix 采用[[mixture-of-experts|混合专家（MoE）]]设计，核心组件如下[^src-dynamix]：

### AL-RNN 专家

每个专家是 Almost-Linear RNN（[[almost-linear-rnn|AL-RNN]]），遵循：

$$z_t = A z_{t-1} + W \Phi^*(z_{t-1}) + h$$

其中 $\Phi^*$ 仅在 P << M 个单元上施加 ReLU 非线性，其余保持线性。前 N 个单元作为读出层提供预测观测 $\hat{x}_t = z_{1:N,t}$。使用 J=10 个专家，每个 M=30，P=2[^src-dynamix]。

### 门控网络

门控网络接收上下文 $C \in \mathbb{R}^{N \times T_C}$ 和当前潜在状态 $z_t$，通过以下步骤生成专家权重：

1. **状态注意力**：基于投影潜在状态与上下文观测之间的距离计算注意力权重 $w_t^{att}$
2. **CNN 编码器**：单层 3 通道 CNN 提取上下文的时间特征 $\tilde{C}$
3. **MLP 门控**：加权 CNN 特征与 $z_t$ 拼接后经两层 MLP + softmax 输出专家权重 $w_t^{exp}$

关键优势：上下文长度灵活可变，不受固定长度限制[^src-dynamix]。

## 训练

- **训练数据**：Gilpin (2022) 的 34 个不同 3D 混沌/周期系统，约 60 万条序列
- **训练方法**：稀疏教师强制（[[sparse-teacher-forcing|STF]]），τ=10
- **损失**：MSE + 方差正则化（λ=0.1）
- **优化器**：RADAM，学习率指数衰减 $5 \times 10^{-3} \to 10^{-5}$
- **总参数量**：约 10k[^src-dynamix]

## 关键能力

1. **零样本外域泛化**：在 54 个未见过的 3D 测试系统上重建正确的长期行为
2. **跨维度泛化**：虽仅训练于 3D 系统，可泛化到 2D（Selkov、Van-der-Pol）和 6D（Lorenz-96）
3. **多变量耦合**：原生处理多变量，捕获维度间的耦合动力学
4. **参数与计算效率**：0.1% 的参数量和数量级更快的推理速度
5. **动力学可解释性**：专家使用模式提供不同系统之间的动力学相似度度量[^src-dynamix]

## 评估指标

- **Dstsp**：状态空间的 KL 散度，衡量吸引子几何一致性
- **DH**：功率谱的 Hellinger 距离，衡量长期时间特性一致性
- **MASE / MAE**：短期 n 步预测误差[^src-dynamix]

## 消融发现

- STF 训练至关重要（普通 BPTT 导致性能崩溃）
- CNN + 注意力机制是关键组件
- AL-RNN > LSTM > 普通 RNN > 储备池计算
- J ≥ 5 个专家足够，更多专家收益递减
- 仅 Lorenz 类训练数据不足以泛化[^src-dynamix]

## 与其他模型对比

| 模型 | DSR 能力 | 参数量 | 零样本 |
|------|----------|--------|--------|
| DynaMix | 完整（几何+时间） | ~10k | 是 |
| [[chronos|Chronos]] | 失败（收敛到不动点/周期） | ~800M | 是（预测） |
| [[timesfm|TimesFM]] | 失败 | 200M | 是（预测） |
| 定制 AL-RNN | 完整 | 极低 | 否（需领域内训练） |

DynaMix 甚至在从未见过的真实世界数据（交通、天气、fMRI、EEG）上超越 [[chronos|Chronos]] 和 [[timesfm|TimesFM]]，尽管后者的训练语料包含类似数据[^src-dynamix]。代码开源：Julia 和 Python 版本。

[^src-dynamix]: [[source-dynamix]]
