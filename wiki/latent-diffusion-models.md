---
title: "Latent Diffusion Models (LDM)"
type: entity
tags:
  - diffusion
  - latent-space
  - text-to-image
  - cvpr-2022
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# Latent Diffusion Models

**潜扩散模型（Latent Diffusion Models, LDM）** 是 Rombach 等人在 CVPR 2022 提出的方法[^src-rombach-ldm-2022]，将扩散模型应用于预训练自编码器的潜空间，而非直接处理像素空间。

## 核心思想

LDM 将图像生成分为两个阶段[^src-rombach-ldm-2022]：

1. **感知压缩阶段**：训练一个自编码器（VQGAN 或 KL-regularized VAE），将图像 x 编码为低维潜变量 z = E(x)，解码器从潜变量重建图像 x̃ = D(z)
2. **语义扩散阶段**：在潜空间上训练扩散模型，学习 p(z)，生成时从潜空间采样后通过解码器得到图像

## 关键设计

### 压缩因子 f

- f = H/h = W/w，表示空间下采样倍数
- f = 1：像素级扩散（LDM-1）
- f = 4-8：最佳权衡（LDM-4, LDM-8）
- f = 16：过度压缩导致信息损失
- f = 32：质量严重下降

### 跨注意力条件化

LDM 通过跨注意力层将条件信息注入 UNet[^src-rombach-ldm-2022]：

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d}}\right)V$$

其中 Q 来自潜变量 z_t，K 和 V 来自条件编码器 τθ(y)。这使得 LDM 可以灵活地接受文本、语义图等多种条件输入。

## 与其他扩散模型的关系

- **与 DDPM**：LDM 沿用 DDPM 的简化训练目标，但作用于潜变量
- **与 EDM**：EDM 是后续工作，专注于像素级扩散的优化；LDM 提供了另一种降低计算成本的方式
- **与 VQGAN**：LDM 的第一阶段使用 VQGAN 架构，但目的不是自回归生成，而是为扩散模型提供高效的潜空间

## 主要成果

- CelebA-HQ：FID 5.11（新 SOTA）
- FFHQ：FID 4.98
- MS-COCO 文本到图像：FID 12.63（CFG）
- ImageNet 类别条件：FID 3.60（LDM-4-G）

## 链接

- [[source-rombach-ldm-2022]] — 论文摘要
- [[diffusion-model]] — 扩散模型基础
- [[ddpm]] — DDPM
- [[classifier-free-guidance]] — 无分类器引导
- [[variational-autoencoder]] — 变分自编码器

[^src-rombach-ldm-2022]: [[source-rombach-ldm-2022]]
