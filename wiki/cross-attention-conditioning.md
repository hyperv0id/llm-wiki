---
title: "Cross-Attention Conditioning"
type: technique
tags:
  - diffusion
  - conditioning
  - text-to-image
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# Cross-Attention Conditioning

**跨注意力条件化**是 LDM 论文中引入的通用条件化机制[^src-rombach-ldm-2022]，通过跨注意力层将任意模态的条件信息注入扩散模型。

## 动机

传统的扩散模型条件化方法（如类别标签、模糊图像）表达能力有限。LDM 希望通过一种通用机制，支持文本、语义图、布局等多种条件输入。

## 架构

跨注意力层的核心公式[^src-rombach-ldm-2022]：

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d}}\right)V$$

其中：
- $Q = W_Q \cdot \phi_i(z_t)$：来自去噪 UNet 的中间表示
- $K = W_K \cdot \tau_\theta(y)$：来自条件编码器
- $V = W_V \cdot \tau_\theta(y)$：来自条件编码器

$\tau_\theta$ 是领域特定的条件编码器，例如：
- 文本条件：BERT tokenizer + Transformer
- 语义图条件：卷积网络

## 训练目标

条件 LDM 的训练目标为[^src-rombach-ldm-2022]：

$$\mathcal{L}_{LDM} = \mathbb{E}_{E(x), y, \varepsilon, t}\left[\|\varepsilon - \theta(z_t, t, \tau_\theta(y))\|^2\right]$$

其中 $\tau_\theta$ 和 $\theta$ 联合优化。

## 应用

LDM 使用跨注意力实现了多种条件生成任务[^src-rombach-ldm-2022]：

1. **文本到图像**：使用 BERT + Transformer 作为 $\tau_\theta$，在 LAION-400M 上训练
2. **语义图到图像**：将语义图下采样后与潜变量拼接
3. **布局到图像**：将边界框信息编码后注入
4. **类别条件**：可视为文本条件的简化版本

## 与其他条件化方法的对比

| 方法 | 灵活性 | 实现复杂度 |
|------|--------|------------|
| 类别标签 | 低 | 简单 |
| 模糊图像 | 中 | 中等 |
| 跨注意力 | 高 | 需要额外编码器 |

## 链接

- [[latent-diffusion-models]] — LDM
- [[classifier-free-guidance]] — 无分类器引导
- [[diffusion-model]] — 扩散模型

[^src-rombach-ldm-2022]: [[source-rombach-ldm-2022]]
