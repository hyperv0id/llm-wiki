---
title: "SMLD (Score Matching Langevin Dynamics)"
type: concept
tags:
  - generative-models
  - score-based-generation
  - langevin-dynamics
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# SMLD

**Score Matching Langevin Dynamics (SMLD)** 是 Yang Song 和 Stefano Ermon 于 2019 年提出的一类生成模型，核心思想是通过**得分匹配**估计数据分布的得分函数（对数密度的梯度），再通过**朗之万动力学**从得分函数中采样生成新样本。[^src-tutorial]

## 工作原理

1. **得分估计**：训练一个神经网络 $s_\theta(x)$ 来近似 $\nabla_x \log p(x)$
2. **朗之万采样**：迭代更新 $x_{t+1} = x_t + \epsilon \nabla_x \log p(x_t) + \sqrt{2\epsilon} z_t$，其中 $z_t \sim \mathcal{N}(0, I)$

## 退火朗之万动力学

单一噪声尺度下的朗之万动力学在低密度区域采样效率低。SMLD 使用**退火**策略：从大噪声尺度开始，逐步减小噪声，每个尺度运行多步朗之万采样。[^src-tutorial]

## 与 DDPM 的关系

| 特性 | SMLD | DDPM |
|------|------|------|
| 前向过程 | 多尺度高斯噪声扰动 | 固定方差高斯噪声 |
| 逆向过程 | 朗之万动力学 | 学习去噪网络 |
| 训练目标 | 得分匹配 | ELBO / 去噪目标 |
| 理论框架 | 朗之万方程 | 变分下界 |

SDE 视角下两者是同一连续时间扩散过程的不同离散化。[^src-tutorial]

## 应用

- 图像生成
- 逆问题求解（去模糊、超分辨率、修复）
- 药物分子生成

## 引用

[^src-tutorial]: [[source-chan-2025-diffusion-tutorial]]
