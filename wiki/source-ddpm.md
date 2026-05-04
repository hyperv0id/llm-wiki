---
title: "Denoising Diffusion Probabilistic Models"
type: source-summary
tags:
  - diffusion-models
  - generative-model
  - nips-2020
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# Denoising Diffusion Probabilistic Models (DDPM)

**Denoising Diffusion Probabilistic Models** 是由 Jonathan Ho, Ajay Jain, Pieter Abbeel 于 2020 年发表的 NeurIPS 论文（arXiv:2006.11239），首次证明了扩散模型能够生成高质量图像样本[^src-ddpm]。

## 核心贡献

### 1. 高质量图像生成

DDPM 在无条件图像生成任务上取得了当时最先进的结果：
- **CIFAR-10**: Inception Score 9.46, FID 3.17
- **LSUN Bedroom**: FID 4.90
- **LSUN Church**: FID 7.89
- **CelebA-HQ 256×256**: 生成质量与 ProgressiveGAN 相当

### 2. 扩散模型与得分匹配的等价性

论文揭示了扩散模型与**去噪得分匹配**之间的深层联系。通过特定的参数化，DDPM 的训练目标等价于在多个噪声水平上进行得分匹配，而采样过程等价于**退火朗之万动力学（Annealed Langevin Dynamics）**的变体[^src-ddpm]。

具体而言，如果将反向过程的均值参数化为：
$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \varepsilon_\theta(x_t, t)
$$

其中 $\varepsilon_\theta$ 预测添加的噪声，则训练目标简化为：
$$
L_{\text{simple}} = \mathbb{E}_{t, x_0, \varepsilon}\left[ \| \varepsilon - \varepsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \varepsilon, t) \|^2 \right]
$$

这与 NCSN 的去噪得分匹配目标高度相似。

### 3. 简化训练目标

论文发现使用简化的训练目标 $L_{\text{simple}}$（仅预测噪声 $\varepsilon$）比使用完整的变分下界（ELBO）能获得更好的样本质量，尽管前者的对数似然略差。这一发现对后续扩散模型的发展产生了深远影响。

### 4. 渐进式解码

DDPM 的采样过程可以解释为一种**渐进式有损解码**：
- 率（Rate）: $L_1 + \cdots + L_T$
- 失真（Distortion）: $L_0$

在采样过程中，图像的低频特征先出现，高频细节后出现，这与自回归解码有本质区别，但可以看作是一种广义的比特排序。

## 技术细节

### 前向过程
$$
q(x_t | x_{t-1}) = \mathcal{N}\left(\sqrt{1 - \beta_t}\, x_{t-1},\; \beta_t I\right)
$$

其中 $\beta_t$ 从 $10^{-4}$ 线性增加到 $0.02$，共 $T = 1000$ 步。

### 反向过程
$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}\left(\mu_\theta(x_t, t),\; \sigma_t^2 I\right)
$$

### 网络架构
使用类似 PixelCNN++ 的 U-Net 架构，配合：
- Group Normalization
- Transformer 位置编码
- 16×16 分辨率的自注意力

## 与前序工作的关系

DDPM 与 [[ncsn|NCSN]]（Song & Ermon, 2019/2020）密切相关：
- 两者都使用多噪声水平的去噪训练
- DDPM 的采样过程可以看作是 Langevin 动力学的一种确定性变体
- DDPM 明确建立了与得分匹配的数学联系，而 NCSN 则隐含地使用了这一思想

## 局限性

- **对数似然不具竞争力**：虽然样本质量高，但 NLL 指标不如其他似然基模型
- **采样速度慢**：需要 1000 步迭代
- **大部分码长用于描述不可感知的细节**：这是扩散模型作为有损压缩器的本质特征

## 引用

[^src-ddpm]: Jonathan Ho, Ajay Jain, Pieter Abbeel. "Denoising Diffusion Probabilistic Models". NeurIPS 2020. arXiv:2006.11239