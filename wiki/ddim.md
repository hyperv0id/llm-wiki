---
title: "DDIM"
type: technique
tags:
  - diffusion-models
  - ddim
  - ode
  - sampling
created: 2026-06-22
last_updated: 2026-06-22
source_count: 3
confidence: high
status: active
---

# DDIM (Denoising Diffusion Implicit Models)

**DDIM** 是 Song 等人于 2021 年提出的扩散模型加速采样方法。其表层贡献是非马尔可夫前向过程的变分推广，但**从 ODE 视角看，DDIM 的本质是 Probability Flow ODE 的一阶 Euler 离散化**[^src-ddim][^src-ddim-ode-spaces-ac-cn]。

## ODE 视角的核心理解

### 为什么 DDIM 可以加速

DDPM 之所以需要 1000 步采样，是因为它将反向过程建模为马尔可夫链的逐步近似。但训练目标 $L_\text{simple}$ 仅依赖于边缘分布 $q(x_t|x_0)$，不依赖转移核 $q(x_t|x_{t-1})$ 的具体形式[^src-ddim]。

从 Fokker-Planck 方程的角度，前向 SDE 的 F-P 方程在 $(f_t, g_t) \to (f_t - \frac{1}{2}(g_t^2 - \sigma_t^2)\nabla\log p_t,\; \sigma_t)$ 的变换下不变[^src-ddim-ode-spaces-ac-cn]。当 $\sigma_t = 0$，SDE 退化为一个**确定性 ODE**——这意味着从 $x_T$ 到 $x_0$ 的映射是唯一的，采样不再需要马尔可夫链的随机游走，而可以在 ODE 轨迹上以任意步长跳步。DDIM 的加速本质上就是 ODE 的大步长离散化[^src-ddim-ode-spaces-ac-cn]。

### DDIM 更新公式的 ODE 解释

DDIM 的确定性更新（$\sigma_t=0$）为[^src-ddim]：

$$x_{t-1} = \sqrt{\alpha_{t-1}}\,\underbrace{\left(\frac{x_t - \sqrt{1-\alpha_t}\,\epsilon_\theta(x_t)}{\sqrt{\alpha_t}}\right)}_{\text{预测 }x_0} + \sqrt{1-\alpha_{t-1}}\cdot\epsilon_\theta(x_t)$$

重新参数化 $\bar{x}_t = x_t/\sqrt{\alpha_t}$，$\sigma(t) = \sqrt{(1-\alpha_t)/\alpha_t}$ 后，这等价于 ODE $d\bar{x} = \epsilon_\theta(\bar{x}/\sqrt{\sigma^2+1})\,d\sigma$ 的 Euler 离散化[^src-ddim]。步长由 $\Delta\sigma = \sigma(t) - \sigma(t-\Delta t)$ 控制——选择子序列 $\tau$ 跳过中间步等价于增大 $\Delta\sigma$。

### 与 Probability Flow ODE 的关系

DDIM 是 Probability Flow ODE 在 VP SDE（$f_t(x) = f_t x$，线性漂移）下的特例[^src-ddim][^src-ddim-ode-spaces-ac-cn]。更一般的 Probability Flow ODE 为 $dx = (f_t(x) - \frac{1}{2}g_t^2\nabla\log p_t)dt$，当 $f_t$ 为非线性时对应更一般的扩散过程（如 VE SDE、sub-VP SDE）。DDIM 论文的 Proposition 1 证明了它与 VE SDE 的概率流 ODE 的等价性。

## DDIM 的独有性质

由于 ODE 的确定性，DDIM 具备随机采样（DDPM）无法实现的能力[^src-ddim]：

**1. 采样一致性**：固定初始噪声 $x_T$，无论用多少步采样，生成图像的高层特征保持一致。$x_T$ 是图像的**确定性隐编码**。

**2. 隐空间插值**：在 $x_T$ 空间做球面线性插值（slerp）可产生语义平滑过渡。这类似于 GAN 的隐空间插值，但 DDPM 由于随机性无法做到。

**3. 可逆编码**：ODE 可正向求解（编码 $x_0 \to x_T$）也可反向求解（解码 $x_T \to x_0$），DDIM 可实现接近无损的重建（1000 步重建误差仅 0.0001 per-dim MSE）[^src-ddim]。这使得 DDIM 类似于一个 Neural ODE / Flow 模型。

## 采样质量与效率

DDIM 在少步采样下显著优于 DDPM[^src-ddim]：

| 步数 | DDIM (η=0) | DDPM (η=1) |
|------|-----------|-----------|
| 10 | 13.36 | 41.07 |
| 20 | 6.84 | 18.36 |
| 50 | 4.67 | 8.01 |
| 100 | 4.16 | 5.78 |
| 1000 | 4.04 | 4.73 |

*CIFAR-10 FID ↓，η=0 为确定性 DDIM，η=1 为随机 DDPM。*

20 步 DDIM 的 FID (6.84) 已接近 1000 步 DDPM (4.73)，实现约 50× 加速。

## 与后续工作的关系

DDIM 开启了 ODE 加速采样的研究方向：

- **[[dpm-solver|DPM-Solver]]** (2022)：利用扩散 ODE 的半线性结构，DDIM 是其**一阶特例**（DPM-Solver-1）；高阶求解器在相同 NFE 下大幅提升质量[^src-dpm-solver]
- **[[rectified-flow|Rectified Flow]]** (2022)：通过迭代 rectification 拉直 ODE 轨迹，使大步长 Euler 更精确
- **[[consistency-models|Consistency Models]]** (2023)：学习 ODE 轨迹上任意点到终点的直接映射，实现一步生成
- **[[instaflow|InstaFlow]]** (2024)：reflow 拉直 PF-ODE 后蒸馏为一步模型

从 DDIM → DPM-Solver → Rectified Flow → Consistency Models，构成了一条清晰的**ODE 精度提升 → 轨迹拉直 → 单步化**的演化路径。

## 引用

[^src-ddim]: [[source-ddim]]
[^src-ddim-ode-spaces-ac-cn]: [[source-ddim-ode-spaces-ac-cn]]
[^src-dpm-solver]: [[source-dpm-solver]]
