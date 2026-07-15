---
title: "DYffusion"
type: entity
tags:
  - diffusion-models
  - spatiotemporal-forecasting
  - probabilistic-forecasting
  - dynamics-forecasting
  - neurips-2023
  - ucsd
created: 2026-06-04
last_updated: 2026-07-16
source_count: 3
confidence: medium
status: active
---

# DYffusion

**DYffusion**（DYnamics-informed Diffusion）是一种面向时空动力学预测的扩散模型框架，由 Cachay 等（UC San Diego）在 NeurIPS 2023 上提出[^src-dyffusion]。其核心创新是将传统扩散模型的高斯加噪-去噪过程替换为时序插值-预测的交互过程，使扩散步与物理时间步直接耦合[^src-dyffusion]。

## 核心设计

### 从加噪到插值：扩散过程的重新定义

传统高斯扩散模型的"前向过程"逐渐破坏数据（加噪），"反向过程"逐步还原（去噪）。DYffusion 将这两个过程替换为：

- **前向过程 = 时序插值**：随机时间条件插值网络 $\mathcal{I}_\phi$ 在 $x_t$ 和 $x_{t+h}$ 之间重建中间快照[^src-dyffusion]
- **反向过程 = 时序预测**：确定性预测网络 $F_\theta$ 始终预测 $x_{t+h}$，输入是越来越接近 $t+h$ 的中间步快照[^src-dyffusion]

整个过程中，模型始终在**数据空间**中操作，从未涉及高斯噪声[^src-dyffusion]。

### 训练：两阶段范式

1. **阶段一**：训练 $\mathcal{I}_\phi$ 最小化 $||\mathcal{I}_\phi(x_t, x_{t+h}, i) - x_{t+i}||^2$，学习插值[^src-dyffusion]
2. **阶段二**：冻结 $\mathcal{I}_\phi$（启用 MC dropout），训练 $F_\theta$ 最小化 $||F_\theta(\mathcal{I}_\phi(x_t, x_{t+h}, i_n), i_n) - x_{t+h}||^2$[^src-dyffusion]

### 推理：Cold Sampling

DYffusion 采用 Cold Sampling 算法（源于 Cold Diffusion），交替执行预测和插值，每步都向前逼近真实 $x_{t+h}$。理论上等价于 Euler 方法求解动力系统 ODE[^src-dyffusion]。

## 关键特性

| 特性 | 说明 |
|------|------|
| **数据空间操作** | 所有中间状态都是可解释的实际物理快照，非噪声[^src-dyffusion] |
| **常数内存** | 训练仅需 $x_t$, $x_{t+i}$, $x_{t+h}$ 三个快照，与预测范围 $h$ 无关[^src-dyffusion] |
| **少步扩散** | 通常 $<50$ 扩散步 vs DDPM $1000$+ 步[^src-dyffusion] |
| **连续时间** | 骨干网络接受任意连续时间输入，支持加速采样[^src-dyffusion] |
| **多步预测** | 中间扩散步结果可直接用作中间时刻的预测[^src-dyffusion] |

## 与相关工作的关系

- **vs 传统高斯扩散**（DDPM、EDM）：DYffusion 不使用高斯噪声，扩散状态保持在数据空间中，效率更高[^src-dyffusion]
- **vs Cold Diffusion**：DYffusion 的 Cold Sampling 算法和"广义扩散模型"框架均源于 Cold Diffusion[^src-dyffusion]
- **vs 视频扩散模型**（MCVD）：MCVD 需要建模完整视频片段，内存 $\propto h$；DYffusion 仅需 3 个快照，内存常数[^src-dyffusion]
- **vs TEDM**：二者都将扩散步与物理时间对齐，但 TEDM 保留了高斯噪声框架，DYffusion 完全替换为插值[^src-dyffusion]
- **vs SimDiff**：SimDiff 是端到端扩散点预测模型，DYffusion 专门针对概率多步动力学预测[^src-dyffusion]
- **vs 时空基础模型**：DYffusion 聚焦于复杂物理系统的动力学演化，时空基础模型（UrbanDiT、Aurora 等）侧重于城市交通/天气等大规模预测[^src-dyffusion]
- **vs [[armd|ARMD]]**：[[armd|ARMD]]（AAAI 2025）与 DYffusion 共享"用确定性变换替代加噪作为扩散步"的思路——DYffusion 用时间插值、ARMD 用序列窗口滑动——但 ARMD 面向单序列 TSF 而非时空动力学预测[^src-armd]
- **vs [[erdm|ERDM]]**：ERDM（NeurIPS 2025）在同一 Navier-Stokes benchmark 上显著超越 DYffusion（~3× CRPS 改进），且 ERDM 使用 EDM 高斯噪声框架 + 滚动窗口，DYffusion 用插值替代噪声。ERDM 的作者与 DYffusion 部分重叠（Cachay & Yu）[^src-erdm]

[^src-dyffusion]: [[source-dyffusion]]
[^src-armd]: [[source-armd]]
[^src-erdm]: [[source-erdm]]
