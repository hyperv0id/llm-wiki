---
title: "Score-based Generative Modeling"
type: concept
tags:
  - generative-model
  - score-matching
  - langevin-dynamics
  - machine-learning
created: 2026-04-28
last_updated: 2026-05-13
source_count: 2
confidence: medium
status: active
---

# Score-based Generative Modeling

**基于分数的生成建模**是一类通过估计数据分布的对数密度梯度（分数 ∇ₓ log p(x)）来构建生成模型的框架[^src-ncsn]。

## 核心思想

传统生成模型直接建模概率密度 p(x) 或学习隐变量生成过程。基于分数的生成建模采用间接路径：

1. **学习分数**：训练神经网络 s_θ(x) ≈ ∇ₓ log p_data(x)
2. **朗之万采样**：使用分数通过朗之万动力学从数据分布中采样

## 理论基础

### 分数定义

分数 (score) 是对数密度函数的梯度：
```
score(x) = ∇ₓ log p(x)
```

分数指向概率密度增长的方向，可理解为在 x 处的概率流方向。

### 朗��万动力学

给定分数函数朗之万动力学通过迭代产生样本：
```
x̃_t = x̃_{t-1} + ε ∇ₓ log p(x̃_{t-1}) + √ε z_t,  z_t ~ N(0, I)
```

当 ε → 0 且 T → ∞ 时，x��_T 收敛至真实分布 p(x) 的样本。

## 面临挑战

### 流形假设

现实数据倾向于集中在高维空间中的低维流形上导致：
- 分数在流形外无定义
- 分数匹配目标在流形上不一致

### 低密度区域

数据稀疏的低密度区域导致：
- 分数估计不准确（缺乏训练信号）
- 朗之万动力学混合慢（模式间跨越困难）

## 解决方案

NCSN 通过以下技术解决上述挑战：

1. **噪声扰动**：用高斯噪声扰动数据使分布充满全空间
2. **多噪声水平**：使用噪声序列 {σᵢ} 从大到小逐步逼近真实分布
3. **退火采样**：从高噪声开始逐步降低噪声水平

## 与其他生成模型的关系

```
基于似然模型 (VAE, Flow, PixelCNN)
        ↓
基于分数模型 (NCSN, DDPM, Score SDE)
        ↓
对抗训练模型 (GANs)
```

## 后续发展

- **DDPM** (Ho & Ermon, 2019)：将分数匹配与扩散过程统一
- **Score-Based SDE** (Song et al., 2021)：用随机微分方程统一框架
- **Diffusion Models** (2022-)：成为主流生成模型
- **Flux Matching** (Pao-Huang et al., 2026)：将分数匹配推广到任意生成向量场，不再限于保守的得分函数[^src-2605-07319]

## 相关页面

- [[ncsn]] — Noise Conditional Score Networks，基于分数生成模型的核心实现
- [[smld]] — SMLD 框架，与 NCSN 关系密切
- [[diffusion-model]] — 扩散模型，与基于分数模型等价
- [[score-matching]] — 分数匹配训练方法
- [[langevin-dynamics]] — 朗之万动力学采样基础
- [[annealed-langevin-dynamics]] — 退火朗之万动力学

## 引用

[^src-ncsn]: [[source-ncsn]]
[^src-2605-07319]: [[source-2605-07319]]