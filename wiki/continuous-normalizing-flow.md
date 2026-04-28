---
title: "Continuous Normalizing Flow (CNF)"
type: concept
tags:
  - normalizing-flow
  - generative-model
  - neural-ode
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# 连续归一化流

**连续归一化流（Continuous Normalizing Flow, CNF）** 是 Neural ODE 论文中提出的生成模型[^src-neural-ode]，基于连续变换的归一化流。

## 离散归一化流

传统归一化流使用变量变换公式[^src-neural-ode]：

$$\log p(z_1) = \log p(z_0) - \log\left|\det\frac{\partial f}{\partial z_0}\right|$$

主要瓶颈是计算行列式 $\det(\partial f/\partial z)$ 复杂度为 O(D³)。

## 瞬时变量变换

CNF 推导出瞬时变量变换公式[^src-neural-ode]：

$$\frac{\partial \log p(z(t))}{\partial t} = -tr\left(\frac{\partial f}{\partial z}(t)\right)$$

只需计算矩阵的迹（trace），复杂度为 O(D)。

## 优势

### 线性宽度成本

迹的线性性使得可以使用多个隐藏单元[^src-neural-ode]：

$$\frac{d\log p(z(t))}{dt} = \sum_{n=1}^{M} tr\left(\frac{\partial f_n}{\partial z}(t)\right)$$

而离散 NF 使用 K 个单隐藏单元层才能达到类似表达能力。

### 可逆性

CNF 前后向变换成本相近[^src-neural-ode]，可以训练最大似然并高效采样。

### 时间依赖动力学

可以参数化流参数为时间的函数 $f(z(t), t)$ 实现时间依赖变换。

## 应用

- 密度估计：Two Circle、Two Moons 数据集
- 生成建模：与 VAE 结合的潜在变量模型

## 链接

- [[neural-ordinary-differential-equation]] — Neural ODE
- [[adjoint-sensitivity-method]] — 伴随灵敏度方法
- [[variational-autoencoder]] — 变分自编码器

[^src-neural-ode]: [[source-neural-ode]]