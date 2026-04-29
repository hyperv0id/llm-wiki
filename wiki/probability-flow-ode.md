---
title: "Probability Flow ODE"
type: technique
tags:
  - diffusion-models
  - sampling
  - likelihood
  - sde
created: 2026-04-28
last_updated: 2026-04-28
source_count: 2
confidence: medium
status: active
---

# Probability Flow ODE

**概率流 ODE (Probability Flow ODE)** 是 Score-Based SDE 论文提出的关键概念。对于每个扩散过程，存在一个确定性 ODE，其轨迹与 SDE 共享相同的边缘概率密度[^src-sde]。

## 数学形式

给定前向 SDE：
$$
dx = f(x, t)dt + g(t)dw
$$

对应的概率流 ODE 为：
$$
dx = \left[f(x, t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)\right]dt
$$

## 核心性质

### 1. 边缘分布等价

对于任意时间 $t$，$x(t)$ 在 ODE 演化下的分布与 SDE 演化下的边缘分布 $p_t(x)$ 相同。

### 2. 精确似然计算

通过瞬时变量变换公式，可以精确计算对数似然：
$$
\log p(x_0) = \log p(x_T) - \int_0^T \text{tr}\left(\frac{d}{dt}\log \pi_t(x(t))\right)dt
$$

其中 $\pi_t$ 是 ODE ��迹上的分布。

### 3. 快速采样

使用黑盒 ODE 求解器（如 RK45）可以显著减少函数评估次数���
- 传统 SDE 采样：~1000 步
- ODE 采样：~100 步（减少 90%+）

### 4. 潜在空间操作

ODE 的确定性允许：
- 数据编码：$x(0) \to z = x(T)$
- 潜在插值：在潜在空间中进行语义编辑
- 可辨识编码：编码由数据分布唯一确定

## 与神经 ODE 的联系

当分数函数 $\nabla_x \log p_t(x)$ 被神经网络 $s_\theta(x, t)$ 近似时，概率流 ODE 成为一个**神经 ODE**：
$$
\frac{dx}{dt} = f(x, t) - \frac{1}{2}g(t)^2 s_\theta(x, t)
$$

## 实验结果

- **CIFAR-10 NLL**: 2.99 bits/dim（使用 sub-VP SDE + ODE）
- **采样效率**：比 SDE 求解器快 10 倍以上

## 局限性

- VE SDE 的 ODE 采样质量显著低于 VP SDE
- 高维数据上 ODE 采样 FID 通常略差于 SDE 采样

## 快速采样进展

- **[[dpm-solver|DPM-Solver]]** (Lu et al., 2022)：利用半线性结构实现约 10 步采样
- **DDIM** (Song et al., 2021)：一阶 ODE 求解器，~50 步
- **RK45** (Song et al., ICLR 2021)：黑盒 ODE 求解器，~60 步

## 引用

[^src-sde]: [[source-sde]]
[^src-dpm-solver]: [[source-dpm-solver]]