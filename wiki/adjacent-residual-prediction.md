---
title: "Adjacent Residual Prediction (ARP)"
type: technique
tags:
  - diffusion-model
  - precipitation-forecasting
  - change-awareness
  - cumulative-error
created: 2026-07-24
last_updated: 2026-07-25
source_count: 1
confidence: medium
status: active
---

# Adjacent Residual Prediction (ARP)

Adjacent Residual Prediction（ARP）是 [[tcp-diffusion|TCP-Diffusion]] 提出的训练目标变换机制，将扩散模型的预测目标从绝对降水值改为相邻时间步的降水变化量（$\Delta$ Rainfall）[^src-tcp]。

## 动机

直接预测绝对降水值对 DL 模型极具挑战——天气系统具有混沌特性，降水强度的绝对量级变化剧烈。NWP 方法通过保持"天气变化"作为预测目标来提升预报的准确性和稳定性[^src-tcp]。ARP 受此启发，将这一理念移植到扩散模型框架中。

## 机制

设 $X^{t}_{rain}$ 为 $t$ 时刻的绝对降水数据，相邻残差定义为：

$$\Delta^t_x = X^{t}_{rain} - X^{t-1}_{rain}$$

模型直接输出未来相邻残差序列 $\hat{D}_y = \{\hat{\Delta}^{n+1}_y, \hat{\Delta}^{n+2}_y, \dots, \hat{\Delta}^{n+m}_y\}$，最终降水预测通过累积获得：

$$\hat{y}_{n+t} = X^{n}_{rain} + \sum_{z=1}^{t} \hat{\Delta}^{n+z}_y$$

## 效果

ARP 赋予模型"变化感知"（change awareness）能力——降水的时空演变与历史观测趋势保持一致，从而[^src-tcp]：

1. **减少累积误差**：预测差值而非绝对值降低了误差在时间步之间的传播
2. **确保物理一致性**：降水变化趋势与观测趋势对齐，避免出现违反物理规律的突变

在 TCP-Diffusion 的消融实验中，将常规降水值预测替换为 ARP 后，ETS 提升约 0.1–15.0%[^src-tcp]。

## 与类似机制的对比

类似机制在 NWP 方法（Kalnay, 2003）和全球天气预报基础模型（如 GenCast、GraphCast 的残差预测）中被广泛采用[^src-tcp]。ARP 的独特之处在于将其嵌入扩散模型的去噪训练框架中——扩散过程直接对残差序列加噪和去噪，而非对绝对降水值。

[^src-tcp]: [[source-tcp]]
