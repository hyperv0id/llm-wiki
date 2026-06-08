---
title: "Non-stationary Diffusion For Probabilistic Time Series Forecasting"
type: source-summary
tags:
  - diffusion-models
  - time-series
  - probabilistic-forecasting
  - ddpm
  - non-stationary
  - icml-2025
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# Source: NsDiff

**作者**: Yifan Li, Xiongxiao Xu, Weiye Wang, Huiyu Li, Cungen Cao, Kai Shu (IIT / Chinese Academy of Sciences)
**发表**: ICML 2025 Spotlight
**领域**: 概率时间序列预测
**代码**: https://github.com/wwy155/NsDiff

## 核心论点

NsDiff 是首个将 Location-Scale Noise Model (LSNM) 整合到扩散概率预测中的工作，将 DDPM 的固定单位方差假设替换为位置-尺度噪声模型 $\mathcal{N}(f_\phi(X), g_\psi(X))$，其中 $f_\phi$ 和 $g_\psi$ 分别估计均值和方差[^src-nsdiff]。论文同时引入不确定性感知噪声调度 (Uncertainty-Aware Noise Schedule, UANS)，将时变方差直接注入扩散过程[^src-nsdiff]。

## 方法

### LSNM
前向过程定义为：
$$q(Y_t | Y_{t-1}) = \mathcal{N}(Y_t; \sqrt{\alpha_t}Y_{t-1} + (1-\alpha_t)f_\phi(X),\; \beta_t^2 g_\psi(X) + \beta_t\alpha_t\sigma_{Y_0})$$
其中 $f_\phi$ 为均值估计器，$g_\psi$ 为方差估计器，$\sigma_{Y_0}$ 为端点方差[^src-nsdiff]。与 DDPM 的 $\mathcal{N}(\sqrt{\alpha_t}Y_{t-1}, \beta_t I)$ 和 TMDM 的 $\mathcal{N}(f_\phi(X), I)$ 相比，LSNM 提供了更灵活的分布建模[^src-nsdiff]。

### 推理
推理时 $\sigma_{Y_0}$ 通过求解 Vieta 二次方程估计（而非直接使用 $g_\psi(X)$），确保逆过程参数可解[^src-nsdiff]。噪声调度 $T=20$ 步，线性 $\beta_1=10^{-4}$ 到 $\beta_T=0.02$[^src-nsdiff]。

### 架构
均值估计器使用 [[non-stationary-transformer|Non-stationary Transformer]]，方差估计器为简单三层 MLP，预训练范式但端到端训练也可行（性能下降仅 1.86%）[^src-nsdiff]。

## 关键结果

在 9 个真实数据集和 2 个合成数据集上与 5 个基线（[[timegrad|TimeGrad]], CSDI, TimeDiff, DiffusionTS, TMDM）比较，NsDiff 实现 SOTA[^src-nsdiff]。QICE 在 ETTh1 下降 47.9%，ETTh2 下降 53.6%，Traffic 下降 66.3%[^src-nsdiff]。不确定性变化最大的 Traffic 数据集（181.83）上改进最显著，证明了方法在非平稳场景下的优势[^src-nsdiff]。

## 贡献

1. 首次将 LSNM 引入概率预测扩散模型，统一了 TMDM（单位方差）等前序工作为特例[^src-nsdiff]
2. UANS 机制使反向过程能动态适应数据不确定性变化[^src-nsdiff]
3. 框架模型无关，均值估计器可替换为更轻量的模型[^src-nsdiff]
4. ICML 2025 Spotlight，计算效率优于前序 SOTA TMDM[^src-nsdiff]

## 消融实验

两个简化变体：(1) w/o LSNM：假设单位方差（等价 TMDM）；(2) w/o UANS：假设完美方差估计器。消融表明 LSNM+UANS 组合对性能和稳定性至关重要[^src-nsdiff]。

[^src-nsdiff]: [[source-nsdiff]]
