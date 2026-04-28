---
title: "噪声条件得分网络 (NCSN)"
type: technique
tags:
  - neural-networks
  - score-based-generation
  - diffusion-models
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# 噪声条件得分网络 (NCSN)

**Noise Conditional Score Network (NCSN)** 是 Yang Song 和 Stefano Ermon 在 SMLD 中提出的架构，用于在多个噪声水平上估计得分函数。[^src-tutorial]

## 核心设计

- 输入：噪声图像 $\tilde{x}$ 和噪声水平索引 $i$
- 输出：$\nabla_{\tilde{x}} \log p_{\sigma_i}(\tilde{x}|x)$ 的估计值
- 通过一个共享网络处理，噪声水平作为条件输入

## 训练目标

多个噪声水平的**加权损失函数**：

$$
\sum_{i=1}^L \lambda(\sigma_i) \mathbb{E}_{p(x)}\mathbb{E}_{\tilde{x} \sim \mathcal{N}(x, \sigma_i^2 I)} \left[ \| s_\theta(\tilde{x}, \sigma_i) + \frac{\tilde{x} - x}{\sigma_i^2} \|^2 \right]
$$

其中 $\lambda(\sigma_i) = \sigma_i^2$ 是典型选择。[^src-tutorial]

## 推断过程

1. 初始化 $x_0 \sim \mathcal{N}(0, \sigma_{\max}^2 I)$
2. 从最大噪声水平到最小噪声水平逐步采样
3. 每个噪声水平运行 $T$ 步朗之万动力学
4. $\sigma_i$ 足够小时，最终样本近似来自数据分布

## 与去噪器的关系

训练好的 NCSN 等价于一个条件去噪器：$(\tilde{x} - \sigma_i^2 s_\theta(\tilde{x}, \sigma_i))$ 是给定噪声水平 $\sigma_i$ 下的去噪估计。[^src-tutorial]

## 引用

[^src-tutorial]: [[source-chan-2025-diffusion-tutorial]]
