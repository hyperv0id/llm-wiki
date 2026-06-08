---
title: "Uncertainty-Aware Noise Schedule (UANS)"
type: technique
tags:
  - diffusion-models
  - noise-scheduling
  - uncertainty
  - ddpm
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Uncertainty-Aware Noise Schedule (UANS)

**Uncertainty-Aware Noise Schedule (UANS)** 是 [[nsdiff|NsDiff]] 提出的噪声调度机制，将 DDPM 的固定方差噪声调度替换为数据感知的时变噪声方差，直接编码进扩散过程[^src-nsdiff]。

## 定义

传统 [[ddpm|DDPM]] 的前向方差为 $\beta_t I$——与数据无关，对所有样本均匀[^src-nsdiff]。UANS 将其扩展为：

$$\sigma_t = \beta_t^2 g_\psi(X) + \beta_t \alpha_t \sigma_{Y_0}$$

其中：
- $g_\psi(X)$：条件方差估计器（3 层 MLP），从历史窗口 $X$ 预测目标窗口的方差[^src-nsdiff]
- $\sigma_{Y_0}$：端点方差
- $\beta_t$：标准 DDPM 噪声调度参数
- $\alpha_t = 1 - \beta_t$

这一设计的直观理解：当 $g_\psi(X)$ 指示数据不确定性较高时，扩散过程会在早期步加入更多噪声（$\beta_t$ 较大），而在后期步保留更精细的结构信息[^src-nsdiff]。

## 推理时的关键挑战

UANS 的核心困难在于推理时 $\sigma_{Y_0}$ 未知——只能用 $g_\psi(X)$ 估计，但 $g_\psi$ 可能存在误差[^src-nsdiff]。NsDiff 的解决方案是：利用去噪过程中预测的 $\sigma_\theta$，通过 Vieta 定理从二次方程 $\lambda_0 \sigma_{Y_0}^2 + \lambda_1 \sigma_{Y_0} + \lambda_2 = 0$ 中反解 $\hat{\sigma}_{Y_0}$，其中 $\lambda_0, \lambda_1, \lambda_2$ 是 $g_\psi(X)$、$\sigma_\theta$、$\alpha_t$、$\beta_t$ 的函数[^src-nsdiff]。

这样，$\sigma_{Y_0}$ 的估计不是仅依赖先验的 $g_\psi(X)$，而是结合了扩散模型在去噪过程中学到的后验信息[^src-nsdiff]。

## 消融实验

与两个简化变体对比（ETTh1 数据集）[^src-nsdiff]：

| 变体 | 前向噪声 | QICE(↓) | CRPS(↓) |
|------|----------|---------|---------|
| w/o LSNM (单位方差) | $\beta_t I$ | 2.821 | 0.452 |
| w/o UANS (完美估计器) | $\beta_t g_\psi(X)$ | 3.184 | 0.413 |
| NsDiff (完整) | $\beta_t^2 g_\psi(X) + \beta_t\alpha_t\sigma_{Y_0}$ | **1.470** | **0.392** |

w/o UANS 虽在 CRPS 上有所改善，但 QICE 反而恶化——说明完全依赖 $g_\psi(X)$ 可能导致过拟合，可控的噪声调度优于纯粹的完美方差估计[^src-nsdiff]。

## 意义

UANS 与 [[location-scale-noise-model|LSNM]] 构成对偶设计——LSNM 定义端点的位置和尺度，UANS 定义前向路径上如何根据端点的不确定性逐步加噪。二者共同使 NsDiff 在高不确定性变化的场景（如 Traffic 数据集，不确定性比=181.83）中表现尤其出色[^src-nsdiff]。

## 参见

- [[nsdiff]] — NsDiff 模型，UANS 的提出者
- [[location-scale-noise-model]] — LSNM，UANS 的对偶概念
- [[diffusion-models]] — 扩散模型概念总览

[^src-nsdiff]: [[source-nsdiff]]
