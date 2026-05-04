---
title: "Score-Based Generative Modeling through Stochastic Differential Equations"
type: source-summary
tags:
  - diffusion-models
  - score-based
  - sde
  - iclr-2021
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# Score-Based Generative Modeling through SDEs

**Score-Based Generative Modeling through Stochastic Differential Equations** 是由 Yang Song, Abhishek Kumar, Jascha Sohl-Dickstein, Stefano Ermon, Diederik P. Kingma, Ben Poole 于 2021 年发表在 ICLR 的里程碑论文（arXiv:2011.13456）。该论文提出了一个统一框架，将 SMLD 和 DDPM 统一在 SDE 视角下[^src-sde]。

## 核心贡献

### 1. SDE 统一框架

论文提出用连续时间 SDE 替代离散噪声扰动：
- **前向 SDE**：将数据分布 $p_0$ 逐渐扩散到先验分布 $p_T$
- **反向 SDE**：从噪声恢复数据，依赖于分数函数 $\nabla_x \log p_t(x)$

### 2. 三种 SDE 变体

| SDE 类型 | 对应方法 | 特点 |
|----------|----------|------|
| **VE SDE** | SMLD (NCSN) | ���差爆炸 |
| **VP SDE** | DDPM | 方差保持 |
| **Sub-VP SDE** | 新提出 | 方差有界，性能更好 |

### 3. 预测-校正采样器 (PC Sampler)

结合数值 SDE 求解器与基于分数的 MCMC 方法：
- **Predictor**：数值 SDE 求解器（如 Euler-Maruyama）
- **Corrector**：Langevin MCMC 或 HMC

### 4. 概率流 ODE

存在一个确定性 ODE 与 SDE 共享相同的边缘概率密度：
$$
dx = \left[f(x, t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)\right]dt
$$

这使得：
- 精确似然计算成为可能
- 快速自适应采样（黑盒 ODE 求解器）
- 潜在空间编码与操作

### 5. 可控生成

条件反向 SDE 可以从无条件分数模型高效估计，实现：
- 类别条件生成
- 图像修复
- 上色
- 其他逆问题

## 实验结果

- **CIFAR-10**: IS 9.89, FID 2.20 (无条件生成新 SOTA)
- **CIFAR-10 NLL**: 2.99 bits/dim (新纪录)
- **1024×1024 图像生成**：首次从基于分数的模型实现

## 与前序工作的关系

- **[[ncsn|NCSN]]** (Song & Ermon, 2019)：VE SDE 的离散化
- **[[ddpm|DDPM]]** (Ho et al., 2020)：VP SDE 的离散化
- SDE 框架将两者统一，并引入新的采样方法和 SDE 变体

## 引用

[^src-sde]: Yang Song, Abhishek Kumar, Jascha Sohl-Dickstein, Stefano Ermon, Diederik P. Kingma, Ben Poole. "Score-Based Generative Modeling through Stochastic Differential Equations". ICLR 2021. arXiv:2011.13456