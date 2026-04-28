---
title: "Heun Sampler"
type: technique
tags:
  - diffusion
  - sampling
  - ode-solver
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# Heun Sampler

Heun 方法（又称 improved Euler 或 trapezoidal rule）是一种二阶 Runge-Kutta ODE 求解器[^src-edm]，在 EDM 论文中被用于替代一阶 Euler 方法进行扩散模型采样。

## 原理

Heun 方法每步进行两次网络评估[^src-edm]：
1. 第一次评估计算 $d_i = dx/dt$ 在 $t_i$
2. 第二次评估计算 $d_i'$ 在 $t_{i+1}$
3. 最终步进：$x_{i+1} = x_i + (t_{i+1} - t_i)(d_i + d_i')/2$

## 误差阶数

- Euler 方法：$O(h^2)$ 局部误差
- Heun 方法：$O(h^3)$ 局部误差（h 为步长）

## 效果

在 CIFAR-10 上[^src-edm]：
- VP 模型：7.3× 加速
- VE 模型：300× 加速
- DDIM 模型：3.2× 加速

## 链接

- [[edm]] — EDM 论文
- [[probability-flow-ode]] — 概率流 ODE
- [[annealed-langevin-dynamics]] — 退火朗之万动力学

[^src-edm]: [[source-edm]]