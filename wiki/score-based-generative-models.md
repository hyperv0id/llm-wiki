---
title: "基于分数的生成模型"
type: concept
tags:
  - generative-model
  - score-function
  - diffusion-model
created: 2026-05-04
last_updated: 2026-05-04
source_count: 0
confidence: medium
status: active
---

# 基于分数的生成模型

**基于分数的生成模型 (Score-Based Generative Models)** 是一类生成模型框架，通过估计数据分布对数密度的梯度（[[score-function|分数函数]] $\nabla_\mathbf{x} \log p(\mathbf{x})$）来生成样本，而无需知道归一化常数。代表性工作包括 [[ncsn|NCSN]]（Song & Ermon, 2020）和 [[score-based-sde|Score-Based SDE]]（ICLR 2021），后者通过 SDE 统一了 SMLD 与 [[ddpm|DDPM]]。

在[[diffusion-model|扩散模型]]的视角下，分数函数与所加噪声直接相关：$\nabla \log p(\mathbf{x}_t) = -\frac{1}{\sqrt{1-\bar{\alpha}_t}} \epsilon_0$，揭示了[[langevin-dynamics|朗之万动力学]]采样与扩散去噪之间的对应关系。

## Related Pages
- [[score-function]]
- [[score-based-generative-modeling]]
- [[ncsn]]
- [[score-based-sde]]
- [[diffusion-model]]