---
title: "分数函数"
type: concept
tags:
  - score-based
  - generative-model
  - diffusion
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

**分数函数**（score function）定义为数据对数概率密度关于数据本身的梯度：$\nabla_{\mathbf{x}} \log p(\mathbf{x})$。它指向对数概率增长最快的方向，描述了数据空间中概率密度的局部几何结构[^src-understanding-diffusion-models]。

## 定义

给定概率密度 $p(\mathbf{x})$，分数函数为：

$$\mathbf{s}(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x})$$

与直接建模概率密度 $p(\mathbf{x})$ 不同，分数函数建模的是密度的**梯度场**。这一转换带来了计算上的关键优势：它避开了归一化常数的计算[^src-understanding-diffusion-models]。

## 与能量模型的关系

在[[energy-based-model|基于能量的模型]]中，概率密度表示为：

$$p(\mathbf{x}) = \frac{1}{Z} \exp(-f(\mathbf{x}))$$

其中 $Z$ 是归一化常数（配分函数），$f(\mathbf{x})$ 是能量函数。此时分数函数为：

$$\nabla_{\mathbf{x}} \log p(\mathbf{x}) = -\nabla_{\mathbf{x}} f(\mathbf{x})$$

分数函数恰好等于能量函数的负梯度，完全不需要计算配分函数 $Z$。这使得分数函数在高维空间中特别有吸引力，因为配分函数的计算通常是不可行的[^src-understanding-diffusion-models]。

## 与扩散模型的联系

通过 [[tweedies-formula|Tweedie 公式]]，扩散模型中加噪数据 $\mathbf{x}_t$ 的分数函数与所加噪声直接相关[^src-understanding-diffusion-models]：

$$\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) = -\frac{\boldsymbol{\epsilon}}{\sqrt{1 - \bar{\alpha}_t}}$$

其中 $\boldsymbol{\epsilon}$ 是添加到干净数据中的高斯噪声，$\bar{\alpha}_t$ 是扩散过程的累积噪声调度参数。这一关系揭示了扩散模型和[[score-based-generative-modeling|基于分数的生成模型]]之间的深层等价性：扩散模型中的去噪过程等价于基于分数的生成模型中的分数匹配过程[^src-understanding-diffusion-models]。

## 分数匹配

[[score-matching|分数匹配]]是一种无需知道真实分数函数即可学习分数模型的训练方法。其优化目标是最小化 Fisher 散度：

$$\mathcal{L} = \mathbb{E}_{p(\mathbf{x})} \left[ \| \mathbf{s}_\theta(\mathbf{x}) - \nabla_{\mathbf{x}} \log p(\mathbf{x}) \|^2 \right]$$

其中 $\mathbf{s}_\theta(\mathbf{x})$ 是由神经网络参数化的分数模型。通过分数匹配，模型可以直接学习数据分布的梯度场，而无需显式建模概率密度[^src-understanding-diffusion-models]。

## Langevin 动力学采样

[[langevin-dynamics|Langevin 动力学]]是一种利用分数函数从概率分布中采样的方法。给定分数函数 $\nabla_{\mathbf{x}} \log p(\mathbf{x})$，采样迭代过程为：

$$\mathbf{x}_{t+1} = \mathbf{x}_t + \frac{\eta}{2} \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) + \sqrt{\eta} \, \mathbf{z}_t$$

其中 $\eta$ 是步长，$\mathbf{z}_t \sim \mathcal{N}(0, \mathbf{I})$ 是随机噪声项。该过程沿着分数函数指示的方向（即概率增长方向）移动，同时注入适量噪声以避免坍缩到局部极大值。当步长足够小且迭代次数足够多时，Langevin 动力学保证收敛到目标分布[^src-understanding-diffusion-models]。

## 关键优势

- **无需归一化常数**：分数函数天然不依赖配分函数 $Z$，适用于高维复杂分布
- **与扩散模型等价**：分数匹配和去噪扩散目标函数在数学上等价，统一了两类生成模型框架
- **灵活的采样**：通过 Langevin 动力学可实现灵活的迭代式采样

[^src-understanding-diffusion-models]: [[source-understanding-diffusion-models]]