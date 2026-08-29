---
title: "Probability Flow ODE"
type: technique
tags:
  - diffusion-models
  - sampling
  - likelihood
  - sde
created: 2026-04-28
last_updated: 2026-08-29
source_count: 4
confidence: medium
status: active
---

# Probability Flow ODE

**概率流 ODE (Probability Flow ODE)** 是 Score-Based SDE 论文提出的关键概念。对于每个扩散过程，存在一个确定性 ODE，其轨迹与 SDE 共享相同的边缘概率密度[^src-sde]。深层原因在于 Fokker-Planck 方程的等价变换：前向 SDE 的 F-P 方程在 $(f_t, g_t) \to (f_t - \frac{1}{2}(g_t^2 - \sigma_t^2)\nabla\log p_t,\; \sigma_t)$ 的变换下保持不变，因此不同的 SDE 可产生相同的边缘分布；当 $\sigma_t=0$ 时 SDE 退化为确定性 ODE，即为概率流 ODE[^src-ddim-ode-spaces-ac-cn]。

## 数学形式

给定前向 SDE：
$$
dx = f(x, t)dt + g(t)dw
$$

对应的概率流 ODE 为：
$$
dx = \left[f(x, t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)\right]dt
$$

## 核心性质

### 1. 边缘分布等价

对于任意时间 $t$，$x(t)$ 在 ODE 演化下的分布与 SDE 演化下的边缘分布 $p_t(x)$ 相同。

### 2. 精确似然计算

通过瞬时变量变换公式，可以精确计算对数似然：
$$
\log p(x_0) = \log p(x_T) - \int_0^T \text{tr}\left(\frac{d}{dt}\log \pi_t(x(t))\right)dt
$$

其中 $\pi_t$ 是 ODE ��迹上的分布。

### 3. 快速采样

使用黑盒 ODE 求解器（如 RK45）可以显著减少函数评估次数���
- 传统 SDE 采样：~1000 步
- ODE 采样：~100 步（减少 90%+）

### 4. 潜在空间操作

ODE 的确定性允许：
- 数据编码：$x(0) \to z = x(T)$
- 潜在插值：在潜在空间中进行语义编辑
- 可辨识编码：编码由数据分布唯一确定

## 与神经 ODE 的联系

当分数函数 $\nabla_x \log p_t(x)$ 被神经网络 $s_\theta(x, t)$ 近似时，概率流 ODE 成为一个**神经 ODE**：
$$
\frac{dx}{dt} = f(x, t) - \frac{1}{2}g(t)^2 s_\theta(x, t)
$$

## 实验结果

- **CIFAR-10 NLL**: 2.99 bits/dim（使用 sub-VP SDE + ODE）
- **采样效率**：比 SDE 求解器快 10 倍以上

## 局限性

- VE SDE 的 ODE 采样质量显著低于 VP SDE
- 高维数据上 ODE 采样 FID 通常略差于 SDE 采样

## DDIM 作为特例

概率流 ODE 在 VP SDE（$f_t(x) = f_t x$，线性漂移）下的特例即为 DDIM[^src-ddim-ode-spaces-ac-cn]。代入对应的参数化关系后，概率流 ODE 化简为：

$$\frac{d}{ds}\left(\frac{x(s)}{\bar{\alpha}(s)}\right) = \epsilon_\theta(x(s), t(s)) \frac{d}{ds}\left(\frac{\bar{\beta}(s)}{\bar{\alpha}(s)}\right)$$

这正是 [[ddim|DDIM]] 的连续形式。DDIM 的加速采样本质上是该 ODE 的大步长 Euler 离散化，DPM-Solver 是它的高阶推广[^src-ddim-ode-spaces-ac-cn]。

## 快速采样进展

- **[[dpm-solver|DPM-Solver]]** (Lu et al., 2022)：利用半线性结构实现约 10 步采样，DDIM 是其**一阶特例**
- **[[ddim|DDIM]]** (Song et al., 2021)：概率流 ODE 在 VP SDE 下的一阶 Euler 离散化，~50 步
- **RK45** (Song et al., ICLR 2021)：黑盒 ODE 求解器，~60 步
- **[[instaflow|InstaFlow]]** (Liu et al., ICLR 2024)：通过 reflow 拉直 PF-ODE 轨迹后蒸馏到一步生成
- **[[swift|Swift]]** (Stock et al., arXiv 2025)：通过 TrigFlow 一致性模型直接单步求解 PF-ODE，每预报步仅需 1 NFE，用于天气预测自回归 rollout，实现 39× 加速[^src-swift]

## 指南口径：作为 FM/GM 特例的概率流 ODE

[[source-flow-matching-guide|FM 指南]]（arXiv:2412.06264，2024-12）第 10 章把扩散采样放进 FM/GM 框架，给出与本页等价但更进一步的口径：

- **时间约定相反**：扩散时间 $r$（数据为 0、噪声为 $+\infty$）与 FM 时间 $t$（噪声为 0、数据为 1）需经严格单调递减映射互换，是避免混淆的纯约定问题[^src-flow-matching-guide]。
- **ODE 采样等价**：在 Gaussian 路径 + 独立耦合 + score 参数化下，概率流 ODE 与 FM 模型的 ODE 采样相同（指南把该等价表述为其设计体系的推论）[^src-flow-matching-guide]。
- **SDE 采样 = 加 divergence-free 分量**：随机 SDE 采样等价于在概率流 ODE 上叠加 divergence-free 的 Langevin 动力学（噪声水平 $\beta_t\ge0$ 为族参数）；指南证明理论上所有 $\beta_t\ge0$ 共享同一边缘，实践中因估计误差与模拟误差存在经验最优 $\beta_t$[^src-flow-matching-guide]。
- **"真"时间反转并非必要**：概率流 ODE 不是扩散过程在 Anderson (1982) 意义下的完整时间反转（后者要求全部有限维分布反转一致），而只共享边缘——对只取终点的生成任务二者无差别；指南转述 Karras et al. (2022) 的定位：概率流 ODE 是低 NFE 采样下的当前最优方案，并引 Ma et al. (2024) 说明"真"时间反转甚至可能次优[^src-flow-matching-guide]。

框架背景见 [[flow-matching]] 与 [[flow-matching-design-space]]。

## 引用

[^src-sde]: [[source-sde]]
[^src-flow-matching-guide]: [[source-flow-matching-guide]]
[^src-swift]: [[source-swift]]
[^src-ddim-ode-spaces-ac-cn]: [[source-ddim-ode-spaces-ac-cn]]