---
title: "LSCD: Lomb–Scargle Conditioned Diffusion for Time Series Imputation"
type: source-summary
tags:
  - diffusion-models
  - time-series
  - data-imputation
  - frequency-domain
  - lomb-scargle
  - icml-2025
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Source: LSCD

**作者**: Elizabeth Fons*, Alejandro Sztrajman*, Yousef El-Laham, Luciana Ferrer, Svitlana Vyetrenko, Manuela Veloso (*Equal contribution; J.P. Morgan AI Research / University of Cambridge / University of Buenos Aires / CONICET)
**发表**: ICML 2025 (PMLR 267)
**arXiv**: 2506.17039
**领域**: 非均匀采样时间序列插补

## 核心论点

LSCD 是首个将可微 Lomb–Scargle 周期图谱估计器集成到扩散模型中的方法，用于非均匀采样或含缺失的时间序列插补[^src-lscd]。问题根源在于：传统频域方法依赖 FFT，要求均匀采样——缺失值必须先插值或填零才能做频谱分析，这在高缺失率下会扭曲频率估计[^src-lscd]。LSCD 的洞察是：Lomb–Scargle 周期图可以**直接从非均匀采样数据中估计功率谱密度**，无需预处理，因此能作为可靠的频率条件信号注入扩散去噪过程[^src-lscd]。

## 方法

LSCD 建立在 [[csdi|CSDI]] 的条件扩散框架之上，增加三个核心组件[^src-lscd]：

1. **可微 Lomb–Scargle 层**：从观测条件 $x_0^{\text{co}}$ 中直接计算 Lomb–Scargle 周期图 $\text{LS}(x_0^{\text{co}}) \in \mathbb{R}^J$，无需插值。频谱经 False Alarm Probability (FAP) 过滤去除伪频率分量，再经对数变换和归一化
2. **基于注意力的频谱编码器 $E_{\text{spec}}$**：两层多头自注意力，编码频率间和特征间的依赖关系，输出嵌入 $z_S$，注入扩散去噪网络的每一步
3. **频谱一致性损失 $L_{\text{SCons}}$**：训练后期精调阶段的损失项 $\|\text{LS}(x_0^{\text{co}}) - \text{LS}(\hat{x}_0^{\text{co}})\|_2^2$，强制最终插补信号与观测信号的频谱对齐

训练分为两阶段：先优化标准得分匹配目标 $L(\theta) = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t^{\text{ta}}, t \mid x_0^{\text{co}}, \text{LS}(x_0^{\text{co}}))\|^2]$，再用 $L_{\text{SCons}}$ 精调以提升频谱保真度[^src-lscd]。

## 关键结果

- **合成正弦波数据**：在 MCAR、序列缺失、块缺失三种机制下，LSCD 的 S-MAE 显著优于所有基线——10% 缺失时 S-MAE 为 0.003 vs CSDI 0.008（↓62.5%），90% 缺失时 0.036 vs CSDI 0.044[^src-lscd]
- **PhysioNet 医疗数据**：10%/50%/90% 缺失率下 MAE 和 S-MAE 均最佳（10%: MAE 0.211 vs CSDI 0.219；90%: MAE 0.479 vs CSDI 0.481）[^src-lscd]
- **PM2.5 空气质量**：MAE 9.069 vs CSDI 9.670（↓6.2%），S-MAE 0.022 vs CSDI 0.023[^src-lscd]
- **消融实验**：删除 LS 条件化导致最大性能退化（PM2.5 MAE 从 9.069 升至 9.669），其次是删除 $E_{\text{spec}}$（9.334），最后是 $L_{\text{SCons}}$（9.085）[^src-lscd]

## 贡献

1. 首次将可微 Lomb–Scargle 引入机器学习，提供可直接集成到学习框架中的实现
2. 证明频谱条件化在极端缺失率（90%）下仍能保持插补质量
3. 理论分析：添加频谱条件信息严格降低扩散反向过程的条件熵 $H(X_{t-1}^{\text{ta}} \mid X_t^{\text{ta}}, X_0^{\text{co}}, Z_S) < H(X_{t-1}^{\text{ta}} \mid X_t^{\text{ta}}, X_0^{\text{co}})$[^src-lscd]
4. 两阶段训练：得分匹配 + 频谱精调，在分布精度和频谱保真度之间寻求平衡

## 局限性

- 架构适配了网格化数据，尽管 Lomb–Scargle 原生支持非均匀采样[^src-lscd]
- 训练时间比 CSDI 增加约 43-45%（含频谱精调阶段），推理增加约 13%[^src-lscd]
- 正弦假设可能引入偏置——频谱一致性损失会偏离得分匹配最优解[^src-lscd]
- 仍依赖固定时间网格（与 CSDI 相同），不适用于连续时间推理[^src-lscd]

[^src-lscd]: [[source-lscd]]
