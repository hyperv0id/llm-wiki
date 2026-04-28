---
title: "朗之万动力学 (Langevin Dynamics)"
type: technique
tags:
  - stochastic-processes
  - sampling
  - diffusion-models
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# 朗之万动力学

朗之万动力学是一类描述粒子在势能场中受随机力驱动的**随机微分方程**，由 Paul Langevin 于 1908 年提出。在生成模型中，它被用作从概率分布中采样的工具。[^src-tutorial]

## 数学形式

朗之万方程的一般形式：

$$
\dot{\xi}(t) + \gamma \xi(t) = \Gamma(t)
$$

其中 $\gamma$ 是阻尼系数，$\Gamma(t)$ 是高斯白噪声，满足 $E[\Gamma(t)] = 0$，$E[\Gamma(t)\Gamma(t')] = q\delta(t-t')$。[^src-tutorial]

用于采样的离散形式：

$$
x_{t+1} = x_t + \epsilon \nabla_x \log p(x_t) + \sqrt{2\epsilon} z_t, \quad z_t \sim \mathcal{N}(0, I)
$$

## 在生成模型中的应用

- **SMLD**：使用朗之万动力学从估计的得分函数中采样
- **退火朗之万动力学**：多尺度逐步采样，解决低密度区域采样困难

## 理论性质

朗之万方程的解是**马尔可夫过程**。当 $t \to \infty$ 时，解 $\xi(t)$ 的分布 $p(x)$ 满足：

$$
p(x) = \sqrt{\frac{\gamma}{2\pi q}} e^{-\frac{\gamma x^2}{2q}}
$$

即零均值高斯分布。这为朗之万采样的收敛性提供了理论保证。[^src-tutorial]

## 引用

[^src-tutorial]: [[source-chan-2025-diffusion-tutorial]]
