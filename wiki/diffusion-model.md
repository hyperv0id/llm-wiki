---
title: "扩散模型"
type: concept
tags:
  - generative-model
  - diffusion
  - vae
  - score-based
created: 2026-04-28
last_updated: 2026-04-28
source_count: 2
confidence: high
status: active
---

# 扩散模型

**扩散模型**是一类生成模型，通过逐步向数据添加噪声（前向过程），然后学习逆转该过程（反向/去噪过程），从而从纯噪声中生成新样本。[^src-understanding-diffusion-models]

## 与变分自编码器的关系

扩散模型可以理解为一种特殊的**马尔可夫层次变分自编码器（Markovian HVAE）**。具体而言，变分扩散模型（VDM）是 Markovian HVAE 在以下三个限制条件下的特例：[^src-understanding-diffusion-models]

1. **潜变量维度等于数据维度**：所有隐变量 $z_t$ 与数据 $x$ 具有相同的维度，不进行维度压缩。
2. **编码器是固定的线性高斯变换**：前向过程不是学习得到的，而是预先定义的高斯转移核，没有可训练参数。
3. **最终潜变量是标准高斯分布**：$p(z_T) = \mathcal{N}(0, I)$，即经过足够多的扩散步后，数据分布被完全破坏为纯噪声。

## 前向过程

前向过程是一个固定的马尔可夫链，逐步向数据添加高斯噪声：[^src-understanding-diffusion-models]

$$
q(x_t | x_{t-1}) = \mathcal{N}\left(\sqrt{\alpha_t}\, x_{t-1},\; (1 - \alpha_t) I\right)
$$

其中 $\alpha_t \in (0, 1)$ 是噪声调度参数。通过重参数化技巧，可以直接从 $x_0$ 一步采样到任意时间步 $t$：

$$
q(x_t | x_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_t}\, x_0,\; (1 - \bar{\alpha}_t) I\right), \quad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
$$

当 $T$ 足够大且 $\bar{\alpha}_T \to 0$ 时，$x_T$ 近似服从标准高斯分布。[^src-understanding-diffusion-models]

## 反向过程

反向过程学习逆转前向加噪过程。模型 $p_\theta(x_{t-1} | x_t)$ 被参数化为高斯分布，其均值由神经网络预测：[^src-understanding-diffusion-models]

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}\left(\mu_\theta(x_t, t),\; \Sigma_\theta(x_t, t)\right)
$$

训练目标是最小化反向过程与真实后验 $q(x_{t-1} | x_t, x_0)$ 之间的 KL 散度，该后验在给定 $x_0$ 的条件下具有闭式解。[^src-understanding-diffusion-models]

## 三种等价的训练目标

扩散模型的训练目标有三种等价的参数化形式，它们对应不同的预测任务：[^src-understanding-diffusion-models]

1. **预测 $x_0$**：直接训练网络从噪声图像 $x_t$ 中恢复原始数据 $x_0$。
2. **预测噪声 $\varepsilon$**：训练网络预测添加到 $x_t$ 中的噪声分量。这是 DDPM 中最常用的形式，对应的简化损失函数为：
   $$
   L_{\text{simple}} = \mathbb{E}_{t, x_0, \varepsilon}\left[ \| \varepsilon - \varepsilon_\theta(x_t, t) \|^2 \right]
   $$
3. **预测得分 $\nabla \log p(x_t)$**：训练网络估计对数概率密度的梯度。

这三种形式通过重参数化相互等价，选择哪种取决于实现便利性和训练稳定性。[^src-understanding-diffusion-models]

## 与基于得分的模型的联系

扩散模型与**基于得分的生成模型**之间存在深刻联系。通过 Tweedie 公式，给定 $x_t$ 时 $x_0$ 的条件期望可以表示为：[^src-understanding-diffusion-models]

$$
\mathbb{E}[x_0 | x_t] = \frac{x_t + (1 - \bar{\alpha}_t) \nabla_{x_t} \log p(x_t)}{\sqrt{\bar{\alpha}_t}}
$$

这表明预测噪声 $\varepsilon_\theta(x_t, t)$ 等价于估计得分函数 $\nabla_{x_t} \log p(x_t)$，两者之间仅差一个常数因子。因此，训练一个去噪扩散模型本质上等价于在多个噪声水平上进行得分匹配。[^src-understanding-diffusion-models]

## 采样过程

采样通过迭代去噪进行，从 $x_T \sim \mathcal{N}(0, I)$ 开始，逐步应用反向转移核：[^src-understanding-diffusion-models]

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \varepsilon_\theta(x_t, t) \right) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)
$$

这一过程可以看作是**朗之万动力学**的离散化模拟——每一步都沿着得分函数的估计方向移动，同时注入适量噪声以保持采样多样性。[^src-understanding-diffusion-models]

## 条件生成

扩散模型支持两种主要的条件生成策略：[^src-understanding-diffusion-models]

- **分类器引导（Classifier Guidance）**：利用一个预训练的分类器 $p(y | x_t)$ 的梯度来引导采样过程，使生成结果符合特定类别 $y$。采样时，在得分估计中加入分类器的对数梯度项。
- **无分类器引导（Classifier-Free Guidance）**：同时训练条件模型 $\varepsilon_\theta(x_t, t, y)$ 和无条件模型 $\varepsilon_\theta(x_t, t, \emptyset)$，在采样时对两者进行插值：
  $$
  \tilde{\varepsilon}_\theta = \varepsilon_\theta(x_t, t, \emptyset) + w \left( \varepsilon_\theta(x_t, t, y) - \varepsilon_\theta(x_t, t, \emptyset) \right)
  $$
  其中 $w \geq 0$ 控制引导强度。无分类器引导是目前最广泛使用的条件生成方法。[^src-understanding-diffusion-models]

## 局限性

扩散模型存在以下主要局限：[^src-understanding-diffusion-models]

- **采样速度慢**：生成一个样本需要数十到数千步的迭代去噪过程，远慢于 GAN 或 VAE 的单步生成。加速方法包括知识蒸馏、快速 ODE 求解器和一致性模型。
- **无压缩的潜变量表示**：由于潜变量维度等于数据维度，扩散模型不提供有意义的低维潜空间，难以进行语义编辑或插值操作。

## 关键实现

- **[[ddpm|DDPM]]**：2020 年 NeurIPS 论文，首次证明扩散模型可生成高质量图像，建立了与得分匹配的等价性[^src-ddpm]
- **[[ncsn]]**：NCSN，DDPM 的重要前身，使用退火朗之万动力学采样

## 引用

[^src-ddpm]: [[source-ddpm]]
[^src-understanding-diffusion-models]: [[source-understanding-diffusion-models]]
