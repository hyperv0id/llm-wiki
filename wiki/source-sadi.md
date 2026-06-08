---
title: "SADI — Self-attention-based Diffusion Model for Time-series Imputation in Partial Blackout Scenarios"
type: source-summary
tags:
  - diffusion-models
  - time-series
  - data-imputation
  - self-attention
  - partial-blackout
  - aaai-2025
created: 2026-06-08
last_updated: 2026-06-08
source_count: 0
confidence: high
status: active
---

# Source: SADI (AAAI 2025)

**SADI** (Self-Attention Diffusion Model for Time Series Imputation) 由 Oregon State University 的 Mohammad Rafid Ul Islam, Prasad Tadepalli & Alan Fern 发表于 AAAI 2025 (arXiv:2503.01737)。

## 核心贡献

SADI 针对此前扩散插补方法（[[csdi|CSDI]]、SSSD）未充分覆盖的"部分停电"(partial blackout)缺失模式——多特征在连续时间步上同时缺失——提出了一种新颖的双阶段自注意力扩散模型。三大贡献：

1. **Partial blackout 缺失模式的定义与系统评估**：将随机缺失、插值、完全停电和预测统一为部分停电的特例，在 4 个真实世界数据集上进行 20 次重复推理实验
2. **显式建模特征依赖与时序相关性**：FDE (Feature Dependency Encoder) 捕获时间感知的特征间依赖，GTA (Gated Temporal Attention) 建模跨特征的时间相关性——两者联合建模替代 CSDI 的分离式背靠背 Transformer
3. **双阶段插补 + 可学习加权组合**：第二阶段 GTA 块精炼第一阶段插补，可学习的动态加权机制 $(1-\tilde{W}_L) \odot \epsilon_1 + \tilde{W}_L \odot \epsilon_2$ 自动平衡两阶段贡献

## 方法概览

### 架构

- **FDE**: 1-D 膨胀卷积（核 1×3）+ self-attention on feature dimension，层间膨胀率递增扩大感受野
- **GTA**: 受 DiffWave/WaveNet 残差块启发，用 self-attention 替代膨胀卷积建模非局部依赖，GLU 激活
- **双阶段**：第二 GTA 块以第一阶段的插补结果 + 原始噪声数据为输入进行精炼
- **加权组合**：$\tilde{W}_L = \text{sigmoid}(\text{linear}(\text{concat}(W_L, M_0^{co})))$ 从注意力权重和缺失掩码中学习

### 训练策略

两种训练策略：
- **SADI-RM**: 纯随机缺失训练（每 epoch 随机掩码目标比例的观测值）
- **SADI-MPB**: 混合训练——先 RM 收敛到良好局部最优，再交替引入 partial blackout 块（随机选择特征数/连续步数/起始位置）

### 推理

50 个样本取均值，T=50 扩散步。

## 实验

4 个数据集：AgAID（葡萄耐寒性，34.41% 天然缺失）、Air Quality（PM2.5，~13% 缺失）、Electricity（370 客户端，无天然缺失）、NACSE（176 气象站温度，23.72% 缺失）。

SADI-MPB 在所有数据集/缺失特征数组合的 MSE 和 CRPS 上全面超越 CSDI、BRITS、SAITS、MICE。SADI 在 Electricity 和 NACSE 等高维数据集上 GPU 显存需求显著低于 CSDI（后者因显存限制被迫减少通道数）。

### 消融实验

三个核心组件均有正面贡献，FDE 对 NACSE 和 Electricity 等高特征相关数据集尤其关键。

## 局限性

- 双阶段架构增加推理计算量
- 依赖显式注意力权重学习加权系数，可能对噪声敏感
- 仅在 time-series imputation 验证，未拓展到 forecasting
- 代码未公开 (anonymous.4open.science)

## 关联页面

- [[sadi]] — SADI 实体页
- [[partial-blackout]] — 部分停电缺失模式
- [[feature-dependency-encoder]] — FDE 技术
- [[gated-temporal-attention]] — GTA 技术
- [[two-stage-imputation]] — 双阶段插补
- [[mixed-partial-blackout-training]] — MPB 训练策略
- [[csdi]] — CSDI，SADI 的直接对比方法
