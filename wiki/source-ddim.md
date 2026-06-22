---
title: "Denoising Diffusion Implicit Models"
type: source-summary
tags:
  - diffusion-models
  - ddim
  - ode
  - iclr-2021
created: 2026-06-22
last_updated: 2026-06-22
source_count: 2
confidence: medium
status: active
---

# Denoising Diffusion Implicit Models (DDIM)

**Denoising Diffusion Implicit Models** 由 Jiaming Song, Chenlin Meng, Stefano Ermon (Stanford) 发表于 ICLR 2021（arXiv:2010.02502）。论文从变分推断角度提出了非马尔可夫前向过程，但其深层意义在于揭示了扩散模型背后的**确定性 ODE 结构**[^src-ddim]。

## 论文的原始论述：非马尔可夫推广

DDIM 的核心观察是：DDPM 的训练目标 $L_\gamma$ 仅依赖于边缘分布 $q(x_t|x_0)$，而非联合分布 $q(x_{1:T}|x_0)$ 的具体形式[^src-ddim]。因此可以构造一族共享相同边缘分布的非马尔可夫前向过程，参数化一个额外的方差向量 $\sigma \in \mathbb{R}^T_{\ge 0}$：

$$q_\sigma(x_{t-1}|x_t, x_0) = \mathcal{N}\left(\sqrt{\alpha_{t-1}}x_0 + \sqrt{1-\alpha_{t-1}-\sigma_t^2}\cdot\frac{x_t-\sqrt{\alpha_t}x_0}{\sqrt{1-\alpha_t}},\; \sigma_t^2 I\right)$$

由此导出的生成过程更新公式为[^src-ddim]：

$$x_{t-1} = \underbrace{\sqrt{\alpha_{t-1}}\left(\frac{x_t - \sqrt{1-\alpha_t}\,\epsilon_\theta(x_t)}{\sqrt{\alpha_t}}\right)}_{\text{预测 }x_0} + \underbrace{\sqrt{1-\alpha_{t-1}-\sigma_t^2}\cdot\epsilon_\theta(x_t)}_{\text{指向 }x_t\text{ 的方向}} + \underbrace{\sigma_t \epsilon}_{\text{随机噪声}}$$

## ODE 视角下的本质

当 $\sigma_t = 0$（即 DDIM 的确定性极限），更新公式退化为[^src-ddim]：

$$\frac{x_{t-1}}{\sqrt{\alpha_{t-1}}} = \frac{x_t}{\sqrt{\alpha_t}} + \left(\sqrt{\frac{1-\alpha_{t-1}}{\alpha_{t-1}}} - \sqrt{\frac{1-\alpha_t}{\alpha_t}}\right) \epsilon_\theta(x_t)$$

引入重新参数化 $\bar{x}(t) = x_t/\sqrt{\alpha_t}$ 和 $\sigma(t) = \sqrt{(1-\alpha_t)/\alpha_t}$，这恰好是以下 ODE 的 Euler 离散化[^src-ddim][^src-ddim-ode-spaces-ac-cn]：

$$\frac{d\bar{x}}{dt} = \epsilon_\theta\left(\frac{\bar{x}}{\sqrt{\sigma^2+1}}, t\right) \frac{d\sigma}{dt}$$

该 ODE 是 Score-Based SDE 框架中 **Probability Flow ODE** 在 VP SDE（线性漂移）下的特例[^src-ddim][^src-ddim-ode-spaces-ac-cn]。DDIM 论文的 Proposition 1 明确证明了这一等价性。

## 从 Fokker-Planck 方程看 DDIM 的本质

苏剑林（2022）从 Fokker-Planck 方程给出了更透彻的推导[^src-ddim-ode-spaces-ac-cn]：前向 SDE $dx = f_t(x)dt + g_t dw$ 对应的 F-P 方程为 $\partial_t p_t = -\nabla\cdot(f_t p_t) + \frac{1}{2}g_t^2\nabla^2 p_t$。该方程在 $f_t \to f_t - \frac{1}{2}(g_t^2 - \sigma_t^2)\nabla\log p_t$、$g_t \to \sigma_t$ 的变换下不变。当 $\sigma_t=0$ 时，SDE 退化为确定性 ODE：$dx = (f_t - \frac{1}{2}g_t^2\nabla\log p_t)dt$。当 $f_t$ 为线性函数时，此 ODE 即为 DDIM。

这揭示了 DDIM 的深层结构：**DDIM 不是一个"加速采样技巧"，而是扩散模型 ODE 本质的必然推论**。

## 关键贡献

1. **加速采样**：通过子序列 $\tau \subset [1,\ldots,T]$ 跳过扩散步，实现 10×–50× 加速，DDIM（$\eta=0$）在少步时始终优于 DDPM
2. **采样一致性**：固定 $x_T$ 后，不同步数生成的样本保留相同的高层特征——$x_T$ 是图像的确定性编码
3. **隐空间插值**：在 $x_T$ 空间做球面插值可实现语义平滑过渡
4. **可逆编码**：ODE 可正向和反向求解，实现 $x_0 \leftrightarrow x_T$ 的编码-解码，重建误差随步数增加而下降

## 实验亮点

- CIFAR-10: DDIM $\eta=0$ 在 20 步取得 FID 6.84（DDPM $\eta=1$ 为 18.36）；100 步 FID 4.16
- CelebA 64×64: 10 步 FID 17.33，100 步 FID 6.53
- 采样 50k 张 CIFAR-10 图像：DDPM 1000 步需 ~20 小时，DDIM 100 步仅需 ~2 小时

## 局限性

- 少步采样时图像细节有损失（FID 随步数减少而增加）
- 确定性采样（$\eta=0$）牺牲了多样性（但换来了可逆性和一致性）
- 论文的 ODE 视角仅在 Section 4.3 简要讨论，未深入展开

## 与后续工作的关系

DDIM 是 DPM-Solver-1（一阶 ODE 求解器）的特例，DPM-Solver 通过高阶格式在相同 NFE 下显著提升质量。从 ODE 视角看，DDIM → DPM-Solver → Rectified Flow → Consistency Models 构成了一条清晰的演化线：离散化精度提升 → 轨迹拉直 → 单步蒸馏。

## 引用

[^src-ddim]: [[source-ddim]]
[^src-ddim-ode-spaces-ac-cn]: [[source-ddim-ode-spaces-ac-cn]]
