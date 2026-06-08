---
title: "LSCD"
type: technique
tags:
  - diffusion-models
  - time-series
  - data-imputation
  - frequency-domain
  - lomb-scargle
  - spectral-conditioning
  - icml-2025
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# LSCD (Lomb–Scargle Conditioned Diffusion)

**LSCD** 是一种基于频谱条件化扩散模型的时间序列插补方法，由 Fons et al. 发表于 ICML 2025[^src-lscd]。其核心创新是将可微 [[lomb-scargle-periodogram|Lomb–Scargle 周期图]]作为频谱条件信号注入条件扩散模型的去噪过程，从而在非均匀采样或含缺失数据的情况下实现更精确的插补和频谱重建[^src-lscd]。

## 问题动机

传统频域时间序列方法依赖 FFT，要求数据均匀采样[^src-lscd]。当数据含缺失值时，必须先用插值或零填充预处理，这在高缺失率下会扭曲真实的频率分布——产生偏移或虚假的频率峰值[^src-lscd]。Lomb–Scargle 周期图可以直接从非均匀采样数据中计算功率谱，无需预处理，因此是更可靠的频谱条件信号来源[^src-lscd]。

## 架构

LSCD 建立在 [[csdi|CSDI]] 的条件得分扩散框架之上，架构包含三个核心组件[^src-lscd]：

### 1. 可微 Lomb–Scargle 层

从观测条件 $x_0^{\text{co}}$ 计算 Lomb–Scargle 周期图 $\text{LS}(x_0^{\text{co}}) \in \mathbb{R}^J$，经 FAP 过滤、对数变换和归一化后作为条件信号。由于该层是纯可微的矩阵运算，可以端到端集成到学习框架中[^src-lscd]。

### 2. 频谱编码器 $E_{\text{spec}}$

两层多头自注意力编码器，捕获频率间（inter-frequency）和特征间（inter-feature）依赖，输出嵌入 $z_S = E_{\text{spec}}(\text{LS}(x_0^{\text{co}}))$[^src-lscd]。$z_S$ 被注入扩散去噪网络 $\epsilon_\theta(x_t^{\text{ta}}, t \mid x_0^{\text{co}}, z_S)$ 的**每一步**，使模型在重建时域信号时可以利用丰富的频域线索[^src-lscd]。

### 3. 频谱一致性损失 $L_{\text{SCons}}$

训练分为两阶段[^src-lscd]：

- **阶段 1**：标准得分匹配 $L(\theta) = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t^{\text{ta}}, t \mid x_0^{\text{co}}, \text{LS}(x_0^{\text{co}}))\|^2]$
- **阶段 2**：引入 [[spectral-consistency-loss|频谱一致性损失]] $L_{\text{SCons}} = \|\text{LS}(x_0^{\text{co}}) - \text{LS}(\hat{x}_0^{\text{co}})\|_2^2$ 精调，强制插补后的信号频谱与观测信号频谱对齐

这相当于正则化损失 $L_{\text{reg}}(\theta) = \lambda_1 L(\theta) + \lambda_2 L_{\text{SCons}}(\theta)$，在分布精度（阶段1）和频谱保真度（阶段2）之间寻求平衡[^src-lscd]。

## 理论基础

LSCD 的理论支撑来自条件熵分析[^src-lscd]：添加频谱条件信息 $Z_S$ 严格降低扩散反向过程的条件熵：

$$H(X_{t-1}^{\text{ta}} \mid X_t^{\text{ta}}, X_0^{\text{co}}, Z_S) < H(X_{t-1}^{\text{ta}} \mid X_t^{\text{ta}}, X_0^{\text{co}})$$

这一结果直接引自 Yang et al. (2024a) 的频率条件扩散理论[^src-lscd]。

## 与 CSDI 的关系

LSCD 直接继承 [[csdi|CSDI]] 的条件扩散框架（自监督掩码训练 + 观测值作为条件 + 双轴注意力），但将 $\epsilon_\theta(x_t^{\text{ta}}, t \mid x_0^{\text{co}})$ 扩展为 $\epsilon_\theta(x_t^{\text{ta}}, t \mid x_0^{\text{co}}, \text{LS}(x_0^{\text{co}}))$，并增加频谱编码器和频谱一致性损失[^src-lscd]。

| 维度 | CSDI (NeurIPS 2021) | LSCD (ICML 2025) |
|------|--------------------|--------------------|
| 频率条件信号 | 无（纯时域） | Lomb–Scargle 周期图 |
| 频谱编码 | 无 | 两层多头自注意力 |
| 训练损失 | 仅 $L_{\text{simple}}$ | $L_{\text{simple}}$ + $L_{\text{SCons}}$ |
| 计算开销 | 基线 | 训练 +43%，推理 +13% |
| PhysioNet 10% MAE | 0.219 | **0.211** |
| Sines MCAR 10% S-MAE | 0.008 | **0.003** |

## 实验配置

- **数据集**：合成正弦波（三种缺失机制：MCAR、序列缺失、块缺失，各三种缺失率 10%/50%/90%）+ PhysioNet（35 特征，48 时间步，~80% 缺失） + 北京 PM2.5（36 特征，36 时间步，~13% 缺失）[^src-lscd]
- **基线**：Mean、Lerp、BRITS、GP-VAE、US-GAN、TimesNet、CSDI、SAITS、ModernTCN[^src-lscd]
- **指标**：MAE（时域）、RMSE（时域）、S-MAE（频谱 MAE，归一化 PSD 间的平均绝对差）[^src-lscd]
- **消融**：逐步删除 $L_{\text{SCons}}$、$E_{\text{spec}}$、LS 条件化[^src-lscd]

## 局限性

- 架构适配固定时间网格，尽管 Lomb–Scargle 原生支持非均匀采样[^src-lscd]
- 两阶段训练增加总训练时间约 43-45%[^src-lscd]
- LS 的正弦假设可能引入偏置——模型在分布精度和频谱保真度之间存在权衡[^src-lscd]
- 与 CSDI 一样依赖固定插值时间点，不适用于需要推理时确定插值点的连续时间场景[^src-lscd]

## 关联页面

- [[source-lscd]] — 源文件摘要
- [[lomb-scargle-periodogram]] — Lomb–Scargle 周期图
- [[spectral-consistency-loss]] — 频谱一致性损失
- [[csdi]] — CSDI，LSCD 的扩散基础框架
- [[source-csdi]] — CSDI 源文件
- [[diffusion-model]] — 扩散模型总览
- [[frequency-aware-conditioning]] — 频率感知条件化
- [[generative-time-series-forecasting]] — 生成式时间序列预测

[^src-lscd]: [[source-lscd]]
