---
title: "Location-Scale Noise Model (LSNM)"
type: concept
tags:
  - diffusion-models
  - non-stationary
  - uncertainty
  - probabilistic-modeling
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Location-Scale Noise Model (LSNM)

**Location-Scale Noise Model (LSNM)** 是一类条件概率模型的数学形式，描述为 $Y = f_\phi(X) + \sqrt{g_\psi(X)} \cdot \varepsilon$，其中 $f_\phi$ 建模位置（均值），$g_\psi$ 建模尺度（方差），$\varepsilon \sim \mathcal{N}(0, I)$ 为随机扰动[^src-nsdiff]。在扩散模型语境下，LSNM 指定了条件 $X$ 下的端点分布为 $\mathcal{N}(f_\phi(X), g_\psi(X))$[^src-nsdiff]。

## 与传统 DDPM 的区别

传统 [[ddpm|DDPM]] 的端点分布为固定的标准正态 $\mathcal{N}(0, I)$，方差 $\beta_t I$ 对所有维度和所有样本相同[^src-nsdiff]。LSNM 将这一假设扩展为：
- **位置 $f_\phi(X)$**：允许去噪终点在数据空间任意位置，而非强制收敛到原点
- **尺度 $g_\psi(X)$**：允许每个样本/维度拥有不同的方差，体现数据的异质性

在 NsDiff 框架下，前向过程变为：

$$q(Y_t | Y_{t-1}) = \mathcal{N}\left(Y_t; \sqrt{\alpha_t}Y_{t-1} + (1-\alpha_t)f_\phi(X),\; \beta_t^2 g_\psi(X) + \beta_t\alpha_t\sigma_{Y_0}\right)$$

其中 $\sigma_{Y_0}$ 为端点方差，$\beta_t^2 g_\psi(X) + \beta_t \alpha_t \sigma_{Y_0}$ 构成 [[uncertainty-aware-noise-schedule|不确定性感知噪声调度]][^src-nsdiff]。

## 意义

LSNM 的核心价值在于**将非平稳性直接编码进扩散模型的先验假设中**[^src-nsdiff]。在时间序列预测场景，历史窗口 $X$ 的条件可以自然传递到未来预测窗口 $Y$ 的位置和尺度——均值估计器捕获趋势和周期模式，方差估计器捕获不确定性水平的变化。这使得模型在训练集与测试集之间存在分布偏移时仍能准确估计不确定性[^src-nsdiff]。

## 与前序工作的关系

在 NsDiff 之前，时序扩散模型可分为两类：
- **$\mathcal{N}(0, I)$ 去噪起点**：[[timegrad|TimeGrad]]、CSDI、TimeDiff、DiffusionTS[^src-nsdiff]
- **$\mathcal{N}(f_\phi(X), I)$ 去噪起点**：TMDM，引入均值估计但方差仍固定为 $I$[^src-nsdiff]

LSNM 将两者统一为特例——TMDM 是 $g_\psi(X)=I$ 的 LSNM，$\mathcal{N}(0,I)$ 去噪是 $f_\phi(X)=0, g_\psi(X)=I$ 的退化形式[^src-nsdiff]。

## 参见

- [[nsdiff]] — NsDiff 模型，LSNM 的主要应用
- [[uncertainty-aware-noise-schedule]] — UANS，LSNM 的对偶设计
- [[diffusion-models]] — 扩散模型概念总览
- [[non-stationary-time-series]] — 非平稳时间序列建模

[^src-nsdiff]: [[source-nsdiff]]
