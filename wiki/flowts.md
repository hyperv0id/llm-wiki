---
title: "FlowTS"
type: entity
tags:
  - rectified-flow
  - flow-matching
  - time-series-generation
  - unconditional-generation
  - conditional-generation
  - adaptive-sampling
  - rope
  - arxiv
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# FlowTS

**FlowTS** 是首个将 rectified flow 应用于时间序列生成的 ODE 模型，来自西湖大学、华盛顿大学、港科大、中科大、KTH 和 UNC Chapel Hill 的联合团队[^src-flowts]。通过概率空间中的直线输运替代迭代式数值求解器，FlowTS 将扩散模型的数百步采样压缩至 30 步，同时提升生成质量[^src-flowts]。

## 核心设计

### Rectified Flow 框架

FlowTS 学习 ODE $\frac{dZ_t}{dt} = v(Z_t, t)$，将噪声 $Z_0 \sim \mathcal{N}(0, I)$ 沿直线路径 $(Z_1 - Z_0)$ 输运到目标分布[^src-flowts]。训练目标为最小二乘回归：

$$\mathcal{L} = \mathbb{E}_{t \sim \text{Logit-Normal}}\left[\|(Z_1 - Z_0) - G(Z_t, t)\|^2\right]$$

其中 $Z_t = t Z_1 + (1-t) Z_0$ 是线性插值[^src-flowts]。与扩散模型的弯曲轨迹（如 DDPM）相比，直线路径理论最优且无需迭代步骤[^src-flowts]。详见 [[rectified-flow-for-time-series|Rectified Flow for TS]]。

### Adaptive Sampling

受强化学习中探索-利用权衡的启发，FlowTS 引入自适应采样策略：使用缩放因子 $t^k$（$k \in (0, 1]$），早期鼓励探索（高噪声水平），后期聚焦利用（密集小步长）[^src-flowts]。$k=1$ 对应均匀采样。随着采样迭代数增加，最优 $k$ 减小[^src-flowts]。详见 [[adaptive-sampling-flow-matching|Adaptive Sampling]]。

### 无条件到条件的无缝适应

FlowTS 的无条件生成模型可以在推理时直接适应条件任务（预测和插补），无需重新训练[^src-flowts]。条件生成通过以下方式实现：在每次迭代中，用已观测值 $Z_1 \odot M$ 替换对应位置的估计值 $\hat{Z}_1$，然后插值到当前时间步进行 ODE 积分[^src-flowts]。

## 架构

基于编码器-解码器 Transformer，包含以下创新组件[^src-flowts]：

| 组件 | 功能 |
|------|------|
| **Trend Synthetic Layers** | 显式捕获长期趋势模式 |
| **Fourier Synthetic Layers** | 捕获周期性和季节模式 |
| **Attention Registers** | 可学习 token 作为持久记忆，聚合全局上下文 |
| **RoPE** | 旋转位置编码，自然衰减特性与时序对齐 |

预测目标分解为残差、均值、趋势和季节性分量，提高可解释性[^src-flowts]。

## 性能 (SOTA)

| 任务 | 指标 | FlowTS | 此前最优 |
|------|------|--------|---------|
| 无条件生成 (Stocks) | Context-FID ↓ | **0.019** | 0.067 |
| 无条件生成 (ETTh) | Context-FID ↓ | **0.011** | 0.061 |
| 无条件长序列 (ETTh-256) | Context-FID ↓ | **0.302** | 0.423 |
| 预测 (Solar 168→24) | MSE ↓ | **213** | 375 |
| 插补 (MuJoCo 70%) | MSE ↓ | **7e-5** | 2.7e-4 |

所有无条件基准（Discriminative、Predictive、Context-FID、Correlational）全面超越 Diffusion-TS、TimeGAN、TimeVAE 等 6 个基线[^src-flowts]。

## 效率

- 仅 30 采样步 + 2,500 训练迭代即可超越 Diffusion-TS 200 步 + 10,000 迭代[^src-flowts]
- 消融确认 RoPE (50k 频率) + 128 attention registers 为最优组合[^src-flowts]
- 无条件训练即可适应条件任务，无需额外条件训练[^src-flowts]

## 局限性

- arXiv 预印本 (v3, Feb 2025)，未经同行评审
- 与 TSFlow (CFM+GP 先验) 不同，FlowTS 使用 rectified flow 的直线输运范式

## 代码

开源：<https://github.com/UNITES-Lab/FlowTS>

## 相关页面

- [[source-flowts]] — 源文件摘要
- [[rectified-flow]] — Rectified Flow 理论基础
- [[rectified-flow-for-time-series]] — Rectified Flow 在时间序列生成中的应用
- [[adaptive-sampling-flow-matching]] — 自适应采样策略
- [[flow-matching]] — Flow Matching 理论基础
- [[generative-time-series-forecasting]] — 生成式时间序列预测
- [[diffusion-models]] — Diffusion Models 理论基础
- [[tsflow]] — TSFlow，首个 CFM 时间序列预测模型 (ICLR 2025)
- [[dits]] — DiTS，Rectified Flow + MM-DiT 双流架构用于 TS 预测 (arXiv 2026)
- [[rope-time-series]] — RoPE 在时间序列中的应用
- [[attention-register-time-series]] — Attention Register 用于时间序列全局建模

[^src-flowts]: [[source-flowts]]
