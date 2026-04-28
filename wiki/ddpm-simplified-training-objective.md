---
title: "DDPM Simplified Training Objective"
type: technique
tags:
  - diffusion-models
  - training-objective
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# DDPM Simplified Training Objective

**DDPM 简化训练目标**（$L_{\text{simple}}$）是 DDPM 论文提出的关键创新，通过让神经网络直接预测添加的噪声 $\varepsilon$ 来简化扩散模型的训练[^src-ddpm]。

## 完整变分下界 vs 简化目标

完整的 ELBO 目标包含多个 KL 散度项：
$$
L = L_T + \sum_{t=1}^{T-1} L_t + L_0
$$

DDPM 发现，使用简化的噪声预测目标可以获得更好的样本质量：
$$
L_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \varepsilon}\left[ \| \varepsilon - \varepsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \varepsilon, t) \|^2 \right]
$$

其中：
- $t \sim \text{Uniform}(\{1, \dots, T\})$
- $x_0 \sim q(x_0)$（真实数据分布）
- $\varepsilon \sim \mathcal{N}(0, I)$（高斯噪声）

## 数学推导

通过重参数化 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \varepsilon$，并利用前向后验的闭式解，可以将原始的 KL 散度目标简化为预测噪声的形式。

## 为什么简化目标效果更好

1. **去噪权重重新分配**：原始目标对不同时间步的去噪任务赋予相同权重，而 $L_{\text{simple}}$ 隐式地给高噪声水平（更大的 $t$）分配更多权重，这些任务更难但对样本质量更重要。

2. **与得分匹配的等价性**：$L_{\text{simple}}$ 等价于在多个噪声水平上进行去噪得分匹配，这正是 NCSN 成功的关键。

3. **训练稳定性**：直接预测噪声比预测均值更稳定，因为噪声的方差在不同时间步相对一致。

## 与其他目标的关系

- **预测 $\varepsilon$**：DDPM 的 $L_{\text{simple}}$
- **预测 $x_0$**：等价的参数化形式
- **预测得分 $\nabla \log p(x_t)$**：通过 Tweedie 公式与预测 $\varepsilon$ 等价

## 引用

[^src-ddpm]: [[source-ddpm]]