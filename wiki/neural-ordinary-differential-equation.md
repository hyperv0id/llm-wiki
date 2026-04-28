---
title: "Neural Ordinary Differential Equation (Neural ODE)"
type: entity
tags:
  - neural-ode
  - continuous-depth
  - ode-solver
  - neurips-2018
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# Neural Ordinary Differential Equation

**神经常微分方程（Neural ODE）** 是 Chen 等人在 NeurIPS 2018 提出的深度学习模型[^src-neural-ode]，用神经网络参数化隐藏状态的导数，通过黑盒 ODE 求解器计算输出。

## 核心思想

传统残差网络定义离散的有限变换序列[^src-neural-ode]：

$$h_{t+1} = h_t + f(h_t, \theta_t)$$

Neural ODE 将其推广为连续变换：

$$\frac{dh(t)}{dt} = f(h(t), t, \theta)$$

从输入层 h(0) 开始，输出层 h(T) 是 ODE 初始值问题在时间 T 的解。

## 主要特性

### 常数内存成本

通过伴随灵敏度方法（adjoint sensitivity method）计算梯度[^src-neural-ode]，无需存储前向传播的中间状态，内存成本为 O(1)。

### 自适应计算

现代 ODE 求解器可以自适应调整评估次数来达到指定精度[^src-neural-ode]，使计算成本随问题复杂度自动缩放。

### 可逆归一化流

连续变换使得变量变换公式更易计算，推导出瞬时变量变换公式[^src-neural-ode]：

$$\frac{\partial \log p(z(t))}{\partial t} = -tr\left(\frac{\partial f}{\partial z}(t)\right)$$

## 与残差网络的关系

残差网络可以看作 Neural ODE 的 Euler 离散化[^src-neural-ode]。随着层数增加、步长减小，两者在极限下等价。

## 主要成果

- MNIST 分类：测试误差 0.42%，参数量 0.22M
- 密度估计：CNF 在 Two Circle/Two Moons 数据集上优于离散 NF
- 时间序列预测：不规则采样螺旋数据 RMSE 显著低于 RNN

## 链接

- [[source-neural-ode]] — 论文摘要
- [[diffusion-model]] — 扩散模型（ODE 在生成模型中的应用）
- [[continuous-normalizing-flow]] — 连续归一化流

[^src-neural-ode]: [[source-neural-ode]]