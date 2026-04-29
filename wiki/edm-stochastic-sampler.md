---
title: "EDM Stochastic Sampler"
type: technique
tags:
  - diffusion
  - sampling
  - stochastic
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# EDM Stochastic Sampler

EDM 随机采样器是 Karras 等人在 2022 年提出的带随机性的采样方法[^src-edm]，通过在采样中间步骤添加/移除噪声来纠正早期采样错误。

## 核心思想

随机采样器在每步执行两个子步骤[^src-edm]：
1. **噪声添加**：$t_i \to \hat{t}_i = t_i + \gamma_i t_i$，其中 $\gamma_i$ 为 churn 因子
2. **ODE 步进**：从 $\hat{t}_i$ 到 $t_{i+1}$ 使用 Heun 方法

## 参数

- $S_{churn}$：控制总体随机性程度
- $S_{tmin}, S_{tmax}$：启用随机性的噪声范围
- $S_{noise}$：新噪声的标准差（通常设为略高于 1）

## 效果

在 CIFAR-10 VP 模型上[^src-edm]：
- 原始训练：随机采样显著优于确定性采样
- EDM 训练：确定性采样反而更好

这表明随着模型质量提升，随机采样的必要性降低。

## 与 ALD 的关系

EDM 随机采样器是 [[annealed-langevin-dynamics]] 的改进版本，核心区别在于：
- ALD 使用固定噪声调度
- EDM 允许更灵活的噪声添加/移除策略

## 链接

- [[edm]] — EDM 论文
- [[heun-sampler]] — Heun 采样器
- [[annealed-langevin-dynamics]] — 退火朗之万动力学
- [[score-based-sde]] — Score-Based SDE

[^src-edm]: [[source-edm]]
