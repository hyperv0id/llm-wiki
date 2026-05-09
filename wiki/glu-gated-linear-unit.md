---
title: "Gated Linear Unit (GLU)"
type: technique
tags:
  - attention
  - feed-forward
  - activation-function
created: 2026-05-04
last_updated: 2026-05-08
source_count: 1
confidence: medium
status: active
---

# Gated Linear Unit (GLU)

**Gated Linear Units (GLU)** are activation-modulated linear transformations where the output is a component-wise product of two linear projections, one of which is gated by a non-linear activation:

$$\text{GLU}(\mathbf{x}) = (\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1) \odot \sigma(\mathbf{x}\mathbf{W}_2 + \mathbf{b}_2)$$

GLU variants (SwiGLU, GEGLU, etc.) replace the sigmoid $\sigma$ with other activations like Swish or GELU. GLUs have become the dominant choice for feed-forward layers in modern Transformer architectures (LLaMA, PaLM, etc.) and are also used as efficient MoE experts in models like [[mixture-of-experts|FaST]].

## 与 Mamba 输入门的关系

Mamba 的输入门 $\mathbf{\Delta}_i$ 本质上是一种 GLU 变体——它对输入 $\mathbf{x}_i$ 进行逐元素门控后再投影到隐藏状态[^src-demystify-mamba-linear-attention-2024]。具体而言：

$$\mathbf{\Delta}_i = \text{Softplus}(\mathbf{x}_i W_1 W_2)$$

这与 GLU 的 $\sigma(\mathbf{x}\mathbf{W}_2 + \mathbf{b}_2) \odot (\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)$ 模式一致，区别在于 Mamba 使用 Softplus 而非 sigmoid 作为门控激活函数，且门控发生在投影之前而非之后。消融实验表明输入门贡献了 +0.9% 的 ImageNet-1K 准确率提升[^src-demystify-mamba-linear-attention-2024]。

## Related Pages
- [[mixture-of-experts]]
- [[gated-linear-units]]

[^src-demystify-mamba-linear-attention-2024]: [[source-demystify-mamba-linear-attention-2024]]