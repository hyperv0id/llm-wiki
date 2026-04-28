---
title: "Score-Based SDE (ICLR 2021)"
type: entity
tags:
  - diffusion-models
  - score-based
  - sde
  - iclr-2021
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# Score-Based SDE

**Score-Based SDE** 是扩散模型领域的里程碑工作，由 Song 等人于 2021 年发表在 ICLR。该论文将 [[ncsn|NCSN]] (SMLD) 和 [[ddpm|DDPM]] 统一在随机微分方程 (SDE) 的框架下[^src-sde]。

## 核心创新

1. **SDE 统一框架**：用连续时间 SDE 替代离散噪声扰动
2. **三种 SDE 变体**：VE SDE、VP SDE、Sub-VP SDE
3. **PC 采样器**：预测-校正框架结合 SDE 求解器与 Langevin MCMC
4. **概率流 ODE**：实现精确似然计算和快速采样
5. **可控生成**：单一无条件模型支持多种条件生成任务

## 技术细节

### 前向 SDE
$$
dx = f(x, t)dt + g(t)dw
$$

### 反向 SDE
$$
dx = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right]dt + g(t)d\bar{w}
$$

### 采样方法
- **Reverse Diffusion Sampler**：与前向过程对称的离散化
- **Predictor-Corrector (PC)**：结合 SDE 求解器与 MCMC
- **Probability Flow ODE**：确��性采样，支持似然计算

## 实验结果

- CIFAR-10: IS 9.89, FID 2.20
- NLL: 2.99 bits/dim
- 首次实现 1024×1024 高质量图像生成

## 相关页面

- [[diffusion-model]] — 扩散模型概念
- [[ncsn]] — NCSN，VE SDE 的离散化
- [[ddpm]] — DDPM，VP SDE 的离散化
- [[predictor-corrector-sampling]] — PC 采样技术
- [[probability-flow-ode]] — 概率流 ODE

## 引用

[^src-sde]: [[source-sde]]