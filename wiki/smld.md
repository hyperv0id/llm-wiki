---
title: "Score-Matching Langevin Dynamics"
type: concept
tags:
  - generative-models
  - diffusion-models
  - score-based-models
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# Score-Matching Langevin Dynamics (SMLD)

SMLD（分数匹配朗之万动力学）是一种生成建模方法，由 Song 和 Ermon 于 2019 年提出[^src-chan-diffusion-tutorial]。与 [[ddpm|DDPM]] 不同，SMLD 直接从数据分布的梯度（分数）出发，通过朗之万动力学进行采样。

> [!info] NCSN 是 SMLD 的主要实现
> [[ncsn]] (Noise Conditional Score Networks) 是 SMLD 框架的核心实现，通过多噪声水平和退火朗之万动力学解决了原始方法的低密度区域估计问题。

## 核心组件

### 1. 朗之万采样方程

给定一个分布 $p(x)$，朗之万方程定义了迭代采样过程[^src-chan-diffusion-tutorial]：

$$
x_{t+1} = x_t + \\tau \\nabla_x \\log p(x_t) + \\sqrt{2\\tau} z, \\quad z \\sim \\mathcal{N}(0, I)
$$

- **梯度上升项** $\\tau \\nabla_x \\log p(x_t)$：将样本推向高概率区域
- **噪声项** $\\sqrt{2\\tau} z$：引入随机性，避免陷入局部极值

> 朗之万方程 = 梯度上升 + 噪声

### 2. Stein 分数函数

分数函数定义为对数密度的梯度[^src-chan-diffusion-tutorial]：

$$
s_\\theta(x) = \\nabla_x \\log p_\\theta(x)
$$

它描述了一个向量场，指示数据点如何在概率密度等高线上移动。在物理中，这等价于"漂移"项。

### 3. 分数匹配训练

由于真实分布 $p(x)$ 未知，需要通过分数匹配学习 $s_\\theta(x)$[^src-chan-diffusion-tutorial]：

| 方法 | 思想 | 优点 | 缺点 |
|------|------|------|------|
| 显式分数匹配 | 核密度估计 $q_h(x)$ 近似 $p(x)$ | 直观 | 高维性能差 |
| 隐式分数匹配 | 使用 Jacobian 迹 | 无需密度估计 | 计算昂贵 |
| **去噪分数匹配** | 添加高斯噪声后预测噪声 | 扩展、高效 | 需要噪声调度 |

**去噪分数匹配 (DSM)** 的损失函数：

$$
J_{\\text{DSM}}(\\theta) = \\mathbb{E}_{p(x)} \\left[ \\frac{1}{2} \\left\\| s_\\theta(x + \\sigma z) + \\frac{z}{\\sigma} \\right\\|^2 \\right]
$$

其中 $z \\sim \\mathcal{N}(0, I)$，$\\sigma$ 为噪声水平。等价于训练一个去噪器：输入带噪图像，输出预测的噪声。

## 噪声条件分数网络 (NCSN)

为了处理多尺度噪声，NCSN 使用多个噪声水平 $\\sigma_1 < \\sigma_2 < \\dots < \\sigma_L$，联合优化[^src-chan-diffusion-tutorial]：

$$
\\mathcal{J}(\\theta) = \\sum_{i=1}^L \\lambda(\\sigma_i) \\cdot \\mathbb{E}_{p(x)} \\left[ \\frac{1}{2} \\left\\| s_\\theta(x + \\sigma_i z) + \\frac{z}{\\sigma_i} \\right\\|^2 \\right]
$$

推理时使用**退火朗之万动力学**：从最大噪声水平开始，逐步减小噪声，在每个水平内运行多步朗之万更新。

## SMLD vs DDPM

| 特性 | SMLD | DDPM |
|------|------|------|
| 前向过程 | 渐进式添加噪声（方差爆炸） | 固定噪声调度（方差保持） |
| 训练目标 | 分数匹配 | 噪声预测（等价于分数匹配） |
| 采样方式 | 朗之万动力学 | 逆向马尔可夫链 |
| 理论基础 | 福克-普朗克方程 | SDE / ELBO |

## 数学基础

SMLD 的稳态分布可以通过[[fokker-planck-equation|福克-普朗克方程]]分析[^src-chan-diffusion-tutorial]：

$$
\\partial_t p(x, t) = -\\partial_x [\\partial_x(\\log p(x)) p(x, t)] + \\partial_x^2 p(x, t)
$$

该方程证明了朗之万动力学收敛到目标分布 $p(x)$。

## 应用

- 图像生成（无条件和条件）
- 逆向问题（去模糊、超分辨率、修复）
- 医学影像重建（MRI、CT）
- 贝叶斯后验采样

## 局限性

- 采样需要数百至数千步，计算成本高
- 分数函数在低密度区域估计不准
- 需要精心设计噪声调度

## 参考文献

- Song & Ermon (2019): *Generative Modeling by Estimating Gradients of the Data Distribution* (NeurIPS) — 即 [[ncsn]]
- Song & Ermon (2020): *Improved Techniques for Training Score-Based Generative Models* (NeurIPS)
- Vincent (2011): *A Connection Between Score Matching and Denoising Autoencoders* (Neural Computation)
- Chan (2024): *Tutorial on Diffusion Models for Imaging and Vision* (arXiv)

---

[^src-chan-diffusion-tutorial]: [[source-chan-diffusion-tutorial]]