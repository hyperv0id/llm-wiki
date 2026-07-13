---
title: "Normalizing Flow"
type: concept
tags:
  - generative-model
  - normalizing-flow
  - probability
created: 2026-04-28
last_updated: 2026-07-13
source_count: 2
confidence: medium
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
| 可逆 1×1 卷积 | $y = Wx$ | $h \cdot w \cdot \log\|\det(W)\|$ |
| ActNorm | $y = s \odot x + b$ | $h \cdot w \cdot \sum \log \|s\|$ |
| 逐通道变换 | 通道重排 | 0 |

## 代表模型

- **NICE** (Dinh et al., 2014): 首个现代归一化流
- **RealNVP** (Dinh et al., 2016): 引入多尺度和耦合层
- **Glow** (Kingma & Dhariwal, 2018): 引入可逆 1×1 卷积

## 时间序列中的条件流

离散归一化流（尤其 RealNVP affine coupling）被用于多变量概率预测：以序列编码器隐状态为条件，将高斯基变换为预测窗联合分布，并可精确优化 log-likelihood[^src-maf]。

- **AR 条件流**：LSTM/Transformer-MAF 等在逐步解码中用流建模输出变异（Rasul et al. 2020 系基线）[^src-maf]。
- **NAR 条件流**：[[manf|MANF]] 用多尺度注意力编码历史，解码器各层条件驱动堆叠 RealNVP，**不回馈**预测窗观测，实现 one-shot 生成并降低相对 $O(D^2T)$ 的串行开销[^src-maf]。
- 与 [[continuous-normalizing-flow|CNF]] / [[flow-matching|Flow Matching]] 路线不同，MANF 属有限层离散双射，似然精确但表达力依赖耦合层深度[^src-maf]。

## 与其他生成模型对比

| 模型 | 优点 | 缺点 |
|------|------|------|
| VAE | 快速采样，潜空间可解释 | ELBO 近似，生成质量有限 |
| GAN | 生成质量高 | 训练不稳定，无编码器 |
| **Flow** | 精确对数似然，可逆编码 | 计算量大，表达能力受限 |
| Diffusion | 生成质量高，理论基础强 | 采样慢 |

## 相关页面

- [[glow]] — Glow 模型
- [[manf]] — 多尺度注意力 + 条件 RealNVP 的时序概率预测
- [[source-maf]] — MANF 源摘要
- [[generative-time-series-forecasting]] — 生成式时序预测谱系
- [[variational-autoencoder]] — VAE
- [[diffusion-model]] — 扩散模型

## 引用

[^src-glow]: [[source-glow]]
[^src-maf]: [[source-maf]]