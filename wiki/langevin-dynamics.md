---
title: "朗之万动力学 (Langevin Dynamics)"
type: technique
tags:
  - stochastic-processes
  - sampling
  - diffusion-models
created: 2026-04-28
last_updated: 2026-07-25
source_count: 2
confidence: medium
status: active
---

# 朗之万动力学

朗之万动力学是一类描述粒子在势能场中受随机力驱动的**随机微分方程**，由 Paul Langevin 于 1908 年提出。在生成模型中，它被用作从概率分布中采样的工具。[^src-chan-2025-diffusion-tutorial]

## 数学形式

朗之万方程的一般形式：

$$
\dot{\xi}(t) + \gamma \xi(t) = \Gamma(t)
$$

其中 $\gamma$ 是阻尼系数，$\Gamma(t)$ 是高斯白噪声，满足 $E[\Gamma(t)] = 0$，$E[\Gamma(t)\Gamma(t')] = q\delta(t-t')$。[^src-chan-2025-diffusion-tutorial]

用于采样的离散形式：

$$
x_{t+1} = x_t + \epsilon \nabla_x \log p(x_t) + \sqrt{2\epsilon} z_t, \quad z_t \sim \mathcal{N}(0, I)
$$

## 在生成模型中的应用

- **SMLD**：使用朗之万动力学从估计的得分函数中采样
- **退火朗之万动力学**：多尺度逐步采样，解决低密度区域采样困难

## 在时间序列扩散模型中的应用

[[timegrad|TimeGrad]]（ICML 2021）的推理过程本质上是退火朗之万动力学：从白噪声 $x_t^N \sim \mathcal{N}(0,I)$ 出发，$N=100$ 步逐步去噪，每一步执行：

$$x_t^{n-1} = \frac{1}{\sqrt{\alpha_n}} \left[ x_t^n - \frac{\beta_n}{\sqrt{1-\bar\alpha_n}} \varepsilon_\theta(x_t^n, h_{t-1}, n) \right] + \sqrt{\tilde\beta_n}\, z$$

其中 RNN 隐状态 $h_{t-1}$ 作为条件信号注入每步去噪[^src-timegrad]。这展示了朗之万采样从纯生成（无条件图像）扩展到条件生成（时序预测）的关键路径：条件信息通过去噪网络 $\varepsilon_\theta$ 在每步间接引导动力学演化方向。

## 理论性质

朗之万方程的解是**马尔可夫过程**。当 $t \to \infty$ 时，解 $\xi(t)$ 的分布 $p(x)$ 满足：

$$
p(x) = \sqrt{\frac{\gamma}{2\pi q}} e^{-\frac{\gamma x^2}{2q}}
$$

即零均值高斯分布。这为朗之万采样的收敛性提供了理论保证。[^src-chan-2025-diffusion-tutorial]
## 引用

[^src-chan-2025-diffusion-tutorial]: [[source-chan-2025-diffusion-tutorial]]
[^src-timegrad]: [[source-timegrad]]
