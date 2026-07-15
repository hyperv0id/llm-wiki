---
title: "TEDM: 时间轴即扩散轴"
type: entity
tags:
  - diffusion-models
  - time-series-forecasting
  - edm
  - score-based-sde
  - iclr-2026
  - multivariate-forecasting
  - real-time-deployment
created: 2026-05-31
last_updated: 2026-07-16
source_count: 2
confidence: medium
status: active
---

# TEDM

TEDM（Time Series Forecasting with Elucidated Diffusion Models）是一个基于扩散模型的多变量时间序列预测框架，由德国航空航天中心（DLR）提出，发表于 ICLR 2026[^src-tedm]。其核心洞察是：将扩散过程的时间轴与物理时间轴对齐，使采样复杂度从 O(SH) 降至 O(H)，实现了实时在线部署的能力。

## 核心创新

### 1. 扩散时间 = 物理时间

传统扩散预测模型在每个物理时间步内需要 S 步扩散采样，总复杂度 O(SH)。TEDM 从理论推导发现差分方程不依赖 dt，因此将扩散时间轴等同于物理时间轴[^src-tedm]。每步 Euler 积分对应一个物理时间步的推进，窗口内所有数据点被网络并行处理[^src-tedm]。结果：H 个 Euler 步完成 H 步预测，复杂度 O(H)。

### 2. 经验估计的 noise/scale schedule

传统 EDM 使用预设 schedule（σt=t, st=1）。TEDM 从理论关系 E(xt)=st·E(x0) 和 Cov(xt)=s²t·Σt 出发，直接从输入窗口数据中经验估计 st 和 Σt[^src-tedm]。这使得 schedule 对每个数据集“定制化”，避免了人工 schedule 带来的归纳偏置。

两种估计策略：
- **累积估计**：用 y1:t 的累积统计量估计 Σt
- **滑动窗口估计**：对局部统计变化更灵活，且避免 t=1 时方差为零的技术问题[^src-tedm]

### 3. 结构化噪声与对角协方差

噪声不再是 i.i.d. 高斯，而是通过 Σt^(1/2) 施加结构化噪声，每个时间步和特征维度可以有不同的噪声水平[^src-tedm]。在实际使用中对角近似，所有矩阵运算退化为逐元素的标量运算，效率极高[^src-tedm]。

### 4. 推广的 EDM 预处理

EDM 的 cskip/cout/cin 预处理方案从标量推广到矩阵形式：
- CΣ;in = (Cov(y)+Σ)^(-1/2)
- CΣ;skip = Cov(y)(Cov(y)+Σ)^(-1)
- c²Σ;out = Cov(y)(Cov(y)+Σ)^(-1)Σ 的最小特征值[^src-tedm]

在对角近似下，这些公式退化为逐维度的独立预处理[^src-tedm]。

## 架构

TEDM 的 denoiser 可采用多种骨干网络[^src-tedm]：
- **LinearNet**：最简单的全连接层 O(Td)，适合资源受限场景
- **UNet**：适配自 ADM 架构，含残差 1D 卷积、自注意力（Kaiser 窗 + RoPE 位置编码）和 alias-free 重采样[^src-tedm]
- **ConvLSTMNet**：卷积 + 双向 LSTM 的混合架构[^src-tedm]

## 性能

| 数据集 | TEDM MSE | 最佳对比方法 | 提升 |
|--------|----------|-------------|------|
| ETTh2 | 0.214 | ARMD 0.311 | 31% |
| ETTm2 | 0.135 | ARMD 0.181 | 25% |
| Exchange | 0.069 | iTransformer 0.086 | 20% |
| Weather | 0.223 | NsDiff 0.223 | tie[^src-tedm] |

训练：每 batch 0.004s，21.3 MB 内存；推理：0.11s，23.9 MB 内存。TEDM 是训练和推理最快的扩散预测方法之一[^src-tedm]。

消融实验显示经验 schedule 相比 st=1 提升高达 85% MSE（66% MAE），且 EDM > iDDPM+DDIM 的结果在时间序列领域同样成立[^src-tedm]。

## 局限性

- 基于 Ito 扩散的框架不适用于长记忆过程、重尾噪声和跳过程[^src-tedm]
- 对角 Σt 近似在高维特征空间（如 Solar 137 维）可能失效[^src-tedm]
- 大振幅变化场景（ETTh1）挑战光滑流假设[^src-tedm]

## 相关页面

### 理论基础
- [[edm]] — EDM 框架，TEDM 的直接前身
- [[score-based-sde]] — Score-Based SDE 统一框架
- [[ddpm]] — DDPM，VP SDE 的离散化
- [[dpm-solver]] — 快速 ODE 求解器

### 对比方法
- [[simdiff]] — SimDiff，端到端扩散点预测 (AAAI 2026)
- [[erdm]] — ERDM，同样基于 EDM 的滚动扩散预测 (NeurIPS 2025)
- [[informer]] — Informer，高效 Transformer (AAAI 2021)
- [[autoformer]] — Autoformer，分解式 Transformer (NeurIPS 2021)
- [[ltsf-linear]] — LTSF-Linear，简单线性 baseline

### 相关概念
- [[diffusion-model]] — 扩散模型总论
- [[edm-design-space]] — EDM 统一设计空间
- [[edm-preconditioning]] — EDM 预处理技术
- [[heun-sampler]] — Heun 采样器
- [[probability-flow-ode]] — 概率流 ODE
- [[annealed-langevin-dynamics]] — 退火朗之万动力学

### 相关技术
- [[structured-noise-for-ts]] — 时间序列的结构化噪声注入
- [[diffusion-physical-time-alignment]] — 扩散-物理时间轴对齐

[^src-tedm]: [[source-tedm]]
