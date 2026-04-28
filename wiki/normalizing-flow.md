---
title: "Normalizing Flow"
type: concept
tags:
  - generative-model
  - normalizing-flow
  - probability
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# Normalizing Flow

**归一化流**（Normalizing Flow）是一类基于可逆变换的生成模型，通过组合多个简单的可逆映射，将简单分布（如高斯分布）转换为复杂的数据分布[^src-glow]。

## 核心思想

给定一个可逆变换 $f: \mathbb{R}^d \to \mathbb{R}^d$，设 $z = f(x)$，则 $x = f^{-1}(z)$。通过变量变换公式：

$$
\log p_\theta(x) = \log p_\theta(z) + \log \left| \det \frac{\partial f^{-1}}{\partial x} \right|
$$

其中 $z \sim p(z)$ 是简单先验（通常为标准高斯分布）。

## 关键性质

1. **可逆性**：$f$ 和 $f^{-1}$ 都易于计算
2. **可计算行列式**：Jacobian 矩阵行列式易于计算
3. **可组合性**：多个简单流可以组合成复杂流

## 常见流变换

| 变换 | 描述 | log-det |
|------|------|---------|
| 仿射耦合层 | $y_a = s \odot x_a + t, y_b = x_b$ | $\sum \log |s|$ |
| 可逆 1×1 ���积 | $y = Wx$ | $h \cdot w \cdot \log\|\det(W)\|$ |
| ActNorm | $y = s \odot x + b$ | $h \cdot w \cdot \sum \log \|s\|$ |
| 逐通道变换 | 通道重排 | 0 |

## 代表模型

- **NICE** (Dinh et al., 2014): 首个现代归一化流
- **RealNVP** (Dinh et al., 2016): 引入多尺度和耦合层
- **Glow** (Kingma & Dhariwal, 2018): 引入可逆 1×1 卷积

## 与其他生成模型对比

| 模型 | 优点 | 缺点 |
|------|------|------|
| VAE | 快速采样，潜空间可解释 | ELBO 近似，生成质量有限 |
| GAN | 生成质量高 | 训练不稳定，无编码器 |
| **Flow** | 精确对数似然，可逆编码 | 计算量大，表达能力受限 |
| Diffusion | 生成质量高，理论基础强 | 采样慢 |

## 相关页面

- [[glow]] — Glow 模型
- [[realnvp]] — RealNVP
- [[variational-autoencoder]] — VAE
- [[gan]] — GAN
- [[diffusion-model]] — 扩散模型

## 引用

[^src-glow]: [[source-glow]]