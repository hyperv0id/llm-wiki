---
title: "Perceptual Compression"
type: concept
tags:
  - compression
  - autoencoder
  - latent-space
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# Perceptual Compression

**感知压缩**是 LDM 论文中提出的概念[^src-rombach-ldm-2022]，将生成建模的压缩过程分为两个层次：感知压缩和语义压缩。

## 两个层次的压缩

### 感知压缩

- 去除高频、不可见的细节
- 保留整体结构和主要特征
- 由自编码器（VAE/VQGAN）完成
- 对应于图像的"纹理"和"噪声"

### 语义压缩

- 学习数据的语义和概念组合
- 由扩散模型在潜空间完成
- 对应于图像的"结构"和"内容"

## 设计原则

LDM 的核心洞察是：扩散模型擅长语义压缩，但不需要处理感知压缩[^src-rombach-ldm-2022]。因此：

1. 将感知压缩交给高效的自编码器
2. 扩散模型专注于语义压缩
3. 两者分离使得各自可以独立优化

## 与其他方法对比

| 方法 | 压缩策略 |
|------|----------|
| 像素级扩散 | 无感知压缩，直接在像素空间 |
| VQGAN | 极端压缩（f=16-32），用自回归模型生成 |
| LDM | 轻度压缩（f=4-8），用扩散模型生成 |

## 实现

LDM 使用两种自编码器正则化[^src-rombach-ldm-2022]：

1. **KL-reg.**：对潜变量施加 KL 散度正则，接近标准高斯分布
2. **VQ-reg.**：使用向量量化层，类似 VQGAN

两者都结合感知损失和对抗损失来确保重建质量。

## 链接

- [[latent-diffusion-models]] — LDM
- [[variational-autoencoder]] — VAE
- [[diffusion-model]] — 扩散模型

[^src-rombach-ldm-2022]: [[source-rombach-ldm-2022]]
