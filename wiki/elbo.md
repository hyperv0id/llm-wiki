---
title: "证据下界"
type: concept
tags:
  - variational-inference
  - vae
  - diffusion
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# 证据下界

**证据下界**（Evidence Lower Bound, ELBO）是潜变量模型中对数边际似然 $\log p(\mathbf{x})$ 的一个下界，因其可计算而广泛用作优化目标。[^src-understanding-diffusion-models]

## 定义与动机

在潜变量模型中，边际似然 $p(\mathbf{x}) = \int p(\mathbf{x}, \mathbf{z}) d\mathbf{z}$ 通常不可解（intractable），因为需要对所有潜变量 $\mathbf{z}$ 进行积分。ELBO 提供了一个可优化的替代目标：最大化 ELBO 等价于在近似后验族中寻找最接近真实后验的分布。[^src-understanding-diffusion-models]

## 推导方式

### 通过 Jensen 不等式

对 $\log p(\mathbf{x})$ 引入近似后验 $q(\mathbf{z}|\mathbf{x})$：

$$
\log p(\mathbf{x}) = \log \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\left[ \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z}|\mathbf{x})} \right] \ge \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\left[ \log \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z}|\mathbf{x})} \right] = \text{ELBO}
$$

其中不等式来自 Jensen 不等式与 $\log$ 的凹性。[^src-understanding-diffusion-models]

### 通过 KL 散度

ELBO 与真实对数似然之间由 KL 散度连接：

$$
\log p(\mathbf{x}) = \text{ELBO} + D_{\text{KL}}\big(q(\mathbf{z}|\mathbf{x}) \parallel p(\mathbf{z}|\mathbf{x})\big)
$$

由于 KL 散度非负，ELBO 始终是 $\log p(\mathbf{x})$ 的下界。当且仅当 $q(\mathbf{z}|\mathbf{x}) = p(\mathbf{z}|\mathbf{x})$ 时等号成立。因此，**最大化 ELBO 等价于最小化近似后验与真实后验之间的 KL 散度**。[^src-understanding-diffusion-models]

## 在 VAE 中的形式

在变分自编码器（Variational Autoencoder, VAE）中，ELBO 被重写为重建项与先验匹配项之差：

$$
\text{ELBO} = \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z})] - D_{\text{KL}}\big(q(\mathbf{z}|\mathbf{x}) \parallel p(\mathbf{z})\big)
$$

- 第一项 $\mathbb{E}_{q}[\log p(\mathbf{x}|\mathbf{z})]$ 是**重建项**，鼓励解码器从潜变量恢复输入。
- 第二项 $-D_{\text{KL}}(q(\mathbf{z}|\mathbf{x}) \parallel p(\mathbf{z}))$ 是**正则项**，鼓励近似后验接近先验 $p(\mathbf{z})$（通常为标准正态分布）。[^src-understanding-diffusion-models]

## 在扩散模型（VDM）中的分解

在变分扩散模型（Variational Diffusion Model, VDM）的框架下，ELBO 可分解为三项之和：

$$
\text{ELBO} = \underbrace{\mathbb{E}_{q(\mathbf{x}_1|\mathbf{x}_0)}[\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)]}_{\text{重建项}} - \underbrace{D_{\text{KL}}(q(\mathbf{x}_T|\mathbf{x}_0) \parallel p(\mathbf{x}_T))}_{\text{先验匹配项}} - \sum_{t=2}^T \underbrace{\mathbb{E}_{q(\mathbf{x}_t|\mathbf{x}_0)}\big[D_{\text{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))\big]}_{\text{去噪匹配项}}
$$

- **重建项**：衡量模型对原始数据的重建能力。
- **先验匹配项**：当 $T$ 足够大时，$q(\mathbf{x}_T|\mathbf{x}_0)$ 趋近于标准正态分布，该项接近于零，通常忽略不计。
- **去噪匹配项**：模型学习反转扩散过程的每一小步——这是扩散模型训练的核心目标。[^src-understanding-diffusion-models]

## 重要性

ELBO 是连接贝叶斯推理与可扩展深度学习的桥梁。它使得以下场景成为可能：[^src-understanding-diffusion-models]

1. **VAE 系列**：使用重参数化技巧和 ELBO 梯度进行端到端训练。
2. **扩散模型**：通过去噪匹配目标的等价形式进行稳定训练。
3. **变分推理**：为任意潜变量模型提供统一的可优化目标。
4. **模型比较**：ELBO 可作为模型证据（marginal likelihood）的代理，用于模型选择和评估。

[^src-understanding-diffusion-models]: [[source-understanding-diffusion-models]]
