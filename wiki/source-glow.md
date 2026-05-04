---
title: "Glow: Generative Flow with Invertible 1×1 Convolutions"
type: source-summary
tags:
  - flow-based-generative-models
  - normalizing-flow
  - openai
  - nips-2018
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# Glow

**Glow** 是由 OpenAI 的 Diederik P. Kingma 和 Prafulla Dhariwal 于 2018 年发表的论文（arXiv:1807.03039），提出了一种基于归一化流的可逆生成模型，通过引入可学习的 1×1 卷积层显著提升了流模型的表达能力[^src-glow]。

## 核心贡献

### 1. 可逆 1×1 卷积

将 RealNVP 中的固定通道置换替换为可学习的 1×1 卷积：
$$
y_{i,j} = W x_{i,j}
$$

其中 $W$ 是 $c \times c$ 的可逆矩阵。使用 LU 分解参数化将计算复杂度从 $O(c^3)$ 降至 $O(c)$：
$$
W = PL(U + \text{diag}(s))
$$

### 2. ActNorm (Activation Normalization)

数据依赖的初始化方法，对每个通道执行仿射变换：
$$
y_{i,j} = s \cdot x_{i,j} + b
$$

参数初始化使得每个通道的激活在初始小批量数据上具有零均值和单位方差。

### 3. 多尺度架构

采用 RealNVP 的多尺度架构，每隔若干层进行下采样（squeeze），减少计算量。

## 实验结果

| 数据集 | bits/dim |
|--------|----------|
| CIFAR-10 | 3.35 |
| ImageNet 32×32 | 4.09 |
| ImageNet 64×64 | 3.81 |
| LSUN Bedroom | 2.38 |
| LSUN Tower | 2.46 |
| CelebA-HQ 256×256 | 0.98 |

## 意义

Glow 是首个能够高效生成高分辨率图像（如 256×256）的基于似然的生成模型，证明了流模型在图像合成领域的潜力。

## 引用

[^src-glow]: Kingma & Dhariwal. "Glow: Generative Flow with Invertible 1×1 Convolutions". NeurIPS 2018. arXiv:1807.03039