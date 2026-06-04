---
title: "DYffusion: A Dynamics-informed Diffusion Model for Spatiotemporal Forecasting"
type: source-summary
tags:
  - diffusion-models
  - spatiotemporal-forecasting
  - probabilistic-forecasting
  - dynamics-forecasting
  - cold-diffusion
  - neurips-2023
created: 2026-06-04
last_updated: 2026-06-04
source_count: 1
confidence: medium
status: active
---

# DYffusion: A Dynamics-informed Diffusion Model for Spatiotemporal Forecasting

**Salva Rühling Cachay, Bo Zhao, Hailey Joren, Rose Yu (UC San Diego), NeurIPS 2023**

## 核心论题

DYffusion 提出一种新型动力学信息扩散模型，将传统扩散模型的高斯加噪-去噪过程重新想象为时序插值-预测交互过程[^src-dyffusion]。核心论点：对于条件预测任务（如动力学预测），将扩散步与物理时间步直接耦合比从高斯噪声中反演更自然、更高效[^src-dyffusion]。

## 方法

### 两阶段训练框架

**阶段一 — 时序插值器（Temporal Interpolator）**：训练随机时间条件插值网络 $\mathcal{I}_\phi$，学习在初始条件 $x_t$ 和目标 $x_{t+h}$ 之间重建中间快照 $x_{t+i}$（$i \in \{1,\dots,h-1\}$）。随机性通过 Monte Carlo dropout 在推理时引入[^src-dyffusion]。

**阶段二 — 预测器（Forecaster / Diffusion Backbone）**：冻结 $\mathcal{I}_\phi$，启用推理随机性，训练确定性预测网络 $F_\theta$，使其从任一中间插值状态 $x_{t+i_n}$ 预测 $x_{t+h}$。即 $F_\theta(\mathcal{I}_\phi(x_t, x_{t+h}, i_n), i_n) \approx x_{t+h}$[^src-dyffusion]。

### 扩散-动力学位移耦合

扩散步 $n$ 通过调度 $S = [i_n]_{n=0}^{N-1}$ 映射到动力学位移 $i_n$，满足 $0 = i_0 < i_n < h$。最简单的调度是 $N = h$ 且 $S = [j]_{j=0}^{h-1}$，即一对一映射。还可使用额外的辅助扩散步（$k > 0$）对应于 $(0, 1)$ 区间内的浮点位移动[^src-dyffusion]。

### Cold Sampling 推理

采样采用 Cold Sampling 算法（从 Cold Diffusion 推广而来），交替执行预测和插值：

1. 从初始条件 $x_t$ 开始
2. $F_\theta$ 预测 $x_{t+h}$ → $\mathcal{I}_\phi$ 插值到下一个中间步
3. 重复，每一步进一步逼近真实 $x_{t+h}$
4. 可选精化步：对每个中间快照再做一次 $x_{t+i} \approx \mathcal{I}_\phi(x_t, F_\theta(x_{t+i_n}, i_n), i)$[^src-dyffusion]

理论上证明了 DYffusion 是一个学习动力系统解的隐式模型，Cold Sampling 等价于 Euler 方法求解该系统的 ODE[^src-dyffusion]。

### 额外技巧

- **一步前瞻损失（One-Step Look-Ahead Loss）**：模拟采样过程中的误差累积，在训练时额外反向传播一步[^src-dyffusion]
- **预测器条件化**：可选的 $x_t$ 条件化（清洁/加噪/不条件化），数据集间最优选择不同[^src-dyffusion]
- **连续时间性质**：$F_\theta$ 和 $\mathcal{I}_\phi$ 均可处理训练中未见过的连续时间输入，实现任意步长采样和加速推理[^src-dyffusion]

## 实验结果

在三个复杂物理系统数据集上评估：[^src-dyffusion]

| 数据集 | 关键结果 |
|--------|---------|
| **SST（海表温度）** | CRPS 0.181 vs Dropout 0.252，DDPM 0.350，MCVD 0.359；推理速度远快于 MCVD（$<50$ 步 vs $1000$ 步） |
| **Navier-Stokes（流体）** | CRPS 0.067 vs Dropout 0.078，MCVD 0.154；OOD 测试差异极小 |
| **Spring Mesh** | 长程稳定性显著优于单步确定性基线；UNet/CNN 基线在长 rollout 中迅速发散 |

消融关键发现：[^src-dyffusion]
- Cold Sampling >> Naive Sampling（SST CRPS 0.181 vs 0.681）
- 插值器推理 dropout 不可或缺（禁用后 SST CRPS 0.320 vs 0.181）
- 辅助扩散步对 SST 显著提升（$k=0$: CRPS 0.208 → $k=25$: 0.181），对 NS 和 SM 无增益
- 预测器条件化（$c(x_t, n)$）的最佳选择因数据集而异

## 计算效率

| 指标 | DYffusion | 标准高斯扩散 |
|------|-----------|-------------|
| 训练内存 | 常数（仅需 $x_t$, $x_{t+i}$, $x_{t+h}$ 三个快照） | $\propto h$（需整段视频） |
| 扩散步数 | $\approx h$（通常 $<50$） | 数百至数千 |
| 推理前向次数 | $3 \times N_2$ | $N_1 \gg N_2$ |

## 贡献

1. 首次提出将物理时间步耦合进扩散模型的动力学预测框架[^src-dyffusion]
2. 显著降低扩散模型在时空预测中的计算与内存需求[^src-dyffusion]
3. 自然支持多步和长程预测、连续时间采样、推理时性能-速度权衡[^src-dyffusion]
4. 理论证明 DYffusion 是隐式动力系统求解器，Cold Sampling 是其 Euler 近似[^src-dyffusion]

## 局限性

- 当前框架假设固定初始条件单步预测，未扩展到完整序列条件[^src-dyffusion]
- 需要在数据集级别调优 dropout 率、辅助扩散步数、预测器条件化方式[^src-dyffusion]
- 仅在中等规模物理系统上验证，未在更大规模真实世界数据集（如天气预测）上测试[^src-dyffusion]

[^src-dyffusion]: [[source-dyffusion]]
