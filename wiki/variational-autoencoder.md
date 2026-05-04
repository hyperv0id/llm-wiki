---
title: "变分自编码器"
type: concept
tags:
  - generative-model
  - vae
  - latent-variable
created: 2026-04-28
last_updated: 2026-05-04
source_count: 2
confidence: medium
status: active
---

# 变分自编码器

**变分自编码器**（Variational Autoencoder, VAE）是一种潜变量生成模型，通过概率编码器-解码器架构学习数据的分布。VAE 将变分推断与神经网络相结合，在最大化[[elbo|证据下界（ELBO）]]的过程中同时学习数据的压缩表示和生成过程。[^src-understanding-diffusion-models]

## 架构

VAE 由两个概率模块组成：[^src-understanding-diffusion-models]

1. **编码器**（Encoder / 推断网络）$q_\phi(\mathbf{z}|\mathbf{x})$：将输入 $\mathbf{x}$ 映射到潜变量 $\mathbf{z}$ 上的一个近似后验分布。通常参数化为对角高斯分布，输出均值 $\mu_\phi(\mathbf{x})$ 和对数方差 $\log\sigma_\phi^2(\mathbf{x})$：
   $$
   q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}\big(\mathbf{z}; \mu_\phi(\mathbf{x}), \text{diag}(\sigma_\phi^2(\mathbf{x}))\big)
   $$

2. **解码器**（Decoder / 生成网络）$p_\theta(\mathbf{x}|\mathbf{z})$：从潜变量 $\mathbf{z}$ 重建数据 $\mathbf{x}$。对于连续数据通常建模为高斯分布，对于二值数据建模为伯努利分布。

编码器与解码器通过**潜变量空间**连接。与标准自编码器不同，VAE 的编码器输出的是概率分布的参数而非确定性的编码，这使得模型可以生成新样本（从先验采样 $\mathbf{z} \sim p(\mathbf{z})$ 后通过解码器生成）。

## 目标函数

VAE 通过最大化[[elbo|证据下界（ELBO）]]来训练：[^src-understanding-diffusion-models]

$$
\text{ELBO} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{\text{KL}}\big(q_\phi(\mathbf{z}|\mathbf{x}) \parallel p(\mathbf{z})\big)
$$

- **重建项** $\mathbb{E}_{q}[\log p_\theta(\mathbf{x}|\mathbf{z})]$：衡量解码器从潜变量恢复原始数据的能力，相当于最小化重建误差。
- **正则项** $D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \parallel p(\mathbf{z}))$：约束近似后验接近先验分布 $p(\mathbf{z})$（通常为标准正态分布 $\mathcal{N}(0, I)$），为潜空间引入结构化先验。

最大化 ELBO 等价于同时最小化重建损失和近似后验与先验之间的 KL 散度。[^src-understanding-diffusion-models]

## 重参数化技巧

VAE 训练中的一个核心挑战是从编码器输出的分布 $q_\phi(\mathbf{z}|\mathbf{x})$ 中采样时，采样操作不可微，导致梯度无法通过随机节点反向传播。**重参数化技巧**（[[reparameterization-trick|Reparameterization Trick]]）解决了这一问题：[^src-understanding-diffusion-models][^src-bluuuuue-reparameterization-trick]

将随机采样 $\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})$ 重写为确定性变换加独立噪声：

$$
\mathbf{z} = \mu_\phi(\mathbf{x}) + \sigma_\phi(\mathbf{x}) \odot \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(0, I)
$$

其中 $\odot$ 表示逐元素乘法。通过这种方式，随机性被转移到独立于模型参数的噪声变量 $\boldsymbol{\varepsilon}$ 上，而 $\mu_\phi$ 和 $\sigma_\phi$ 的梯度可以正常传播，使 VAE 能够通过标准反向传播进行端到端的随机梯度优化。[^src-understanding-diffusion-models]

重参数化不仅打通了梯度路径，还显著降低了梯度估计方差——相比 REINFORCE 的得分函数估计，重参数化在训练初期可将方差降低数量级（Xu et al., 2019, AISTATS）。[^src-bluuuuue-reparameterization-trick]

## 与扩散模型的关系

变分自编码器与变分扩散模型（VDM）之间存在深刻的统一关系。在马尔可夫层次变分自编码器（Markovian HVAE）框架下，VDM 引入了三个关键限制（潜变量维度等于数据维度、编码器为固定线性高斯变换、最终潜变量为标准高斯分布）。当层次深度 $T=1$ 时，VDM 退化回标准的 VAE：[^src-understanding-diffusion-models]

- **VAE（$T=1$）**：单层潜变量，编码器与解码器均为可学习神经网络，潜变量维度可低于数据维度（有损压缩）。
- **VDM（$T>1$）**：多层潜变量构成马尔可夫链，每一层的编码器为固定高斯核，潜变量维度等于数据维度（无压缩）。

因此，VAE 可以视为扩散模型在单步潜变量变换下的特例，而扩散模型则是 VAE 在深度层次化方向上的推广。这一视角将 VAE 的 ELBO 目标与扩散模型的去噪匹配目标统一在同一个数学框架下。[^src-understanding-diffusion-models]

## 局限性

VAE 的主要局限包括：[^src-understanding-diffusion-models]

- **生成质量**：由于 ELBO 是一个下界而非精确似然，且 KL 正则项会迫使潜变量分布趋向先验，VAE 生成的样本往往比 GAN 或扩散模型模糊。
- **后验坍塌**：当解码器能力过强时，模型可能忽略潜变量 $\mathbf{z}$，导致 $q_\phi(\mathbf{z}|\mathbf{x})$ 退化为先验 $p(\mathbf{z})$，丧失有意义的隐表示。$\beta$-VAE 等变体通过调整 KL 项权重缓解该问题。
- **先验假设**：标准 VAE 假设潜变量服从各向同性高斯分布，这限制了其拟合复杂多模态分布的能力。后续工作如 VampPrior、Normalizing Flow 增强先验等尝试解决此问题。

[^src-understanding-diffusion-models]: [[source-understanding-diffusion-models]]
[^src-bluuuuue-reparameterization-trick]: [[source-bluuuuue-reparameterization-trick]]
