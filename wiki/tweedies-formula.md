---
title: "Tweedie 公式"
type: concept
tags:
  - statistics
  - score-function
  - diffusion-model
created: 2026-05-04
last_updated: 2026-05-04
source_count: 0
confidence: medium
status: active
---

# Tweedie 公式

**Tweedie 公式 (Tweedie's Formula)** 是一个统计学结果，描述了带有加性高斯噪声的随机变量的后验期望与其边际密度分数函数的关系：

$$\mathbb{E}[\mathbf{x} | \mathbf{x}_t] = \mathbf{x}_t + \sigma_t^2 \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)$$

在[[diffusion-model|扩散模型]]中，Tweedie 公式建立了加噪数据 $\mathbf{x}_t$ 的[[score-function|分数函数]]与干净数据 $\mathbf{x}_0$ 的 MMSE 估计之间的联系，是去噪得分匹配（DSM）和 DDPM 简化训练目标的理论基础。

## Related Pages
- [[score-function]]
- [[diffusion-model]]
- [[ddpm-simplified-training-objective]]