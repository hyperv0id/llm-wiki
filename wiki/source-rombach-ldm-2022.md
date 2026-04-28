---
title: "High-Resolution Image Synthesis with Latent Diffusion Models"
type: source-summary
tags:
  - diffusion
  - latent-space
  - text-to-image
  - cvpr-2022
created: 2026-04-28
last_updated: 2026-04-28
source_count: 0
confidence: high
status: active
---

# High-Resolution Image Synthesis with Latent Diffusion Models

**论文信息**：Rombach et al., CVPR 2022

## 核心贡献

LDM（Latent Diffusion Models）提出将扩散模型应用于预训练自编码器的潜空间，而非直接处理像素空间[^src-rombach-ldm-2022]。主要贡献包括：

1. **感知压缩**：使用 VQGAN 或 KL-regularized VAE 将图像压缩到低维潜空间（压缩因子 f=4-16），在保持感知等价的同时大幅降低计算成本
2. **潜空间扩散**：在压缩后的潜空间上训练扩散模型，训练速度提升 2.7×，FID 改进 1.6×
3. **跨注意力条件化**：通过跨注意力层将条件编码器（如 BERT-transformer）的输出注入 UNet，实现文本到图像、语义图到图像等多种条件生成

## 主要成果

- CelebA-HQ：FID 5.11（新 SOTA）
- MS-COCO 文本到图像：FID 12.63（250 DDIM steps, CFG s=1.5）
- ImageNet 类别条件：FID 3.60（LDM-4-G, CFG）
- 超分辨率、去噪、图像修复等任务达到 SOTA

## 局限性

- 顺序采样过程仍比 GAN 慢
- 重建能力可能成为需要精细像素精度的任务的瓶颈

## 源文件
[^src-rombach-ldm-2022]: [[raw/ldm]]