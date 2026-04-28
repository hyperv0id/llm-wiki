---
title: "Glow"
type: entity
tags:
  - flow-based-generative-models
  - normalizing-flow
  - openai
  - nips-2018
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# Glow

**Glow** 是 OpenAI 于 2018 年发布的基于归一化流（Normalizing Flow）的生成模型，由 Diederik P. Kingma 和 Prafulla Dhariwal 提出[^src-glow]。该模型通过引入可逆 1×1 卷积层，显著提升了流模型的表达能力，成为流模型发展的重要里程碑。

## 核心创新

1. **可逆 1×1 卷积**：用可学习的卷积替换 RealNVP 中的固定通道置换，增强模型表达能力
2. **ActNorm**：数据依赖的激活归一化，解决小批量训练时的性能下降问题
3. **多尺度架构**：采用 RealNVP 的多尺度设计，平衡计算效率与生成质量

## 技术细节

### 定义

归一化流通过可逆变换将数据分布映射到潜在空间：
$$
z \sim \mathcal{N}(0, I), \quad x = g_\theta(z)
$$

对数似然通过变量变换公式计算：
$$
\log p_\theta(x) = \log p_\theta(z) + \log \left| \det \frac{\partial z}{\partial x} \right|
$$

### 可逆 1×1 卷积

设输入张量形状为 $h \times w \times c$，权重矩阵 $W \in \mathbb{R}^{c \times c}$，则：
$$
y_{i,j} = W x_{i,j}
$$

对数行列式：
$$
\log \left| \det \frac{\partial \text{conv2D}(x; W)}{\partial x} \right| = h \cdot w \cdot \log |\det(W)|
$$

使用 LU 分解参数化：
$$
W = PL(U + \text{diag}(s)) \quad \Rightarrow \quad \log |\det(W)| = \sum \log |s_i|
$$

### ActNorm

$$
y = s \cdot x + b
$$

初始化时，使每个通道的激活具有零均值和单位方差。

## 实验结果

| 数据集 | bits/dim |
|--------|----------|
| CIFAR-10 | 3.35 |
| ImageNet 32×32 | 4.09 |
| ImageNet 64×64 | 3.81 |
| LSUN Bedroom | 2.38 |
| CelebA-HQ 256×256 | 0.98 |

## 相关页面

- [[normalizing-flow]] — 归一化流概念
- [[realnvp]] — RealNVP，前身工作
- [[nice]] — NICE，早期流模型
- [[variational-autoencoder]] — VAE，对比生成模型
- [[gan]] — GAN，对比生成模型

## 引用

[^src-glow]: [[source-glow]]