---
title: SNR-t Bias
type: concept
tags:
  - diffusion-model
  - bias
  - signal-to-noise-ratio
  - cvpr-2026
created: 2026-05-09
last_updated: 2026-05-09
source_count: 1
confidence: high
status: active
---

# SNR-t Bias

**SNR-t Bias（信噪比-时间步偏置）** 是扩散模型在推理阶段出现的一种系统性偏置：预测样本的实际信噪比（SNR）与其被指派的去噪时间步 $t$ 之间发生错配，导致每一步的去噪网络 $\epsilon_\theta(\cdot, t)$ 接收到与训练时分布不匹配的输入，从而产生系统性预测误差。[^src-2604.16044]

## 定义

在训练阶段，每个时间步 $t$ 的样本 SNR 严格确定：

$$\text{SNR}(t) = \frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}$$

网络 $\epsilon_\theta(x_t, t)$ 学会处理具有该特定 SNR 的输入。但在推理阶段，由于两种误差的累积——（1）神经网络自身的预测误差、（2）数值求解器（ODE/SDE 求解器）的离散化误差——逆过程轨迹偏离理想路径。这使得预测样本 $\hat{x}_t$ 的实际 SNR 不再等于 $\text{SNR}(t)$，而系统性地偏低。[^src-2604.16044]

核心表现：在同一时间步 $t$，$\|\epsilon_\theta(\hat{x}_t, t)\|_2$ 始终大于 $\|\epsilon_\theta(x_t, t)\|_2$（Key Finding 2），说明模型在处理 $\hat{x}_t$ 时将其视为 SNR **更低**的样本。

## 与 Exposure Bias 的关系

Exposure bias 关注的是训练和推理之间样本分布的偏移（$x_t$ 与 $\hat{x}_t$ 的差异），属于 inter-sample bias。SNR-t bias 则关注样本与时间步之间的错配。[^src-2604.16044]

两者的关系：**SNR-t bias 是更根本的偏置，可以诱发 exposure bias**。SNR 错配导致每一步的预测误差，这些误差累积起来造成分布偏移。本文的 Key Finding 1（滑动窗口实验）直接揭示了 SNR 错配如何导致网络系统性高估或低估输出，这是 exposure bias 研究未能触及的解释层面。

## 理论证明（Theorem 5.1）

基于假设 $x_\theta^0(\hat{x}_t, t) = \gamma_t x_0 + \phi_t \epsilon_t$（$0 < \gamma_t \leq 1$），可推导出逆过程样本 $\hat{x}_t$ 的实际 SNR：

$$\text{SNR}_{\text{reverse}}(t) = \frac{\hat{\gamma}^2_t \bar{\alpha}_t}{1 - \bar{\alpha}_t + \left( \frac{\sqrt{\bar{\alpha}_t} \beta_{t+1}}{1-\bar{\alpha}_{t+1}} \phi_{t+1} \right)^2}$$

对比 $\text{SNR}_{\text{forward}}(t) = \bar{\alpha}_t/(1-\bar{\alpha}_t)$，分母多出正项 $(\cdot)^2$，因此 $\text{SNR}_{\text{reverse}}(t) < \text{SNR}_{\text{forward}}(t)$，严格证明了偏置的存在。[^src-2604.16044]

## 去偏方法

本文提出 [[dcw|DCW（Differential Correction in Wavelet Domain）]] 来缓解 SNR-t bias。核心思想：从每次去噪步骤的差分信号 $\hat{x}_{t-1} - x_\theta^0(\hat{x}_t, t)$ 中提取指向理想样本的梯度方向信息，在小波域对各频率子带分别修正。[^src-2604.16044]

## 意义

SNR-t bias 的发现为理解和改进扩散模型提供了新的视角。相比 exposure bias 的 inter-sample 层面分析，SNR-t bias 深入到 sample-timestep 层面，揭示了更根本的误差来源，并且有完整的理论证明支持。这一概念可以推广到任何基于时间步条件化的生成模型中。

[^src-2604.16044]: [[source-snr-t-bias]]
