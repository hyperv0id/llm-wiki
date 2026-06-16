---
title: "Hopf-Cole Transform"
type: technique
tags:
  - schrödinger-bridge
  - stochastic-processes
  - optimal-control
  - partial-differential-equations
created: 2026-06-16
last_updated: 2026-06-16
source_count: 1
confidence: high
status: active
---

# Hopf-Cole Transform

Hopf-Cole transform 是将 [[schrodinger-bridge|Schrödinger bridge]] 问题的 **nonlinear HJB-FP optimality system 线性化**的关键数学工具。通过变量变换，它将耦合的非线性 PDE 系统（Hamilton-Jacobi-Bellman + Fokker-Planck）转化为一对关于 Schrödinger potentials $(\varphi_t, \hat{\varphi}_t)$ 的**线性 PDE**[^src-schrodinger-bridges-generative-modeling]。

## 变量变换

变换定义为：

$$\psi_t = \log\varphi_t, \quad p_t^\star = \varphi_t\hat{\varphi}_t$$

其中 $\psi_t$ 是 optimal cost-to-go（forward potential 的负对数），$p_t^\star$ 是 optimal marginal density，$\varphi_t, \hat{\varphi}_t$ 是 forward 和 backward Schrödinger potentials。

## 原始 Nonlinear System

变换前，SB optimality conditions 构成耦合的 HJB-FP 系统：

$$\begin{aligned}
\partial_t \psi_t + \frac{\sigma_t^2}{2}\|\nabla\psi_t\|^2 + \langle\nabla\psi_t, f\rangle &= -\frac{\sigma_t^2}{2}\Delta\psi_t \\
\partial_t p_t^\star + \nabla\cdot(p_t^\star(f + \sigma_t^2\nabla\psi_t)) &= \frac{\sigma_t^2}{2}\Delta p_t^\star
\end{aligned}$$

HJB 方程（上）控制 optimal control policy；FP 方程（下）控制该 policy 下密度的演化。两者都包含非线性耦合项 $\sigma_t^2 \nabla \psi_t$。

## 变换后：Linearized Hopf-Cole PDEs

代入 $\psi_t = \log\varphi_t$ 并利用 density factorization，得到线性系统：

$$\begin{aligned}
\partial_t \varphi_t + \langle\nabla\varphi_t, f\rangle &= -\frac{\sigma_t^2}{2}\Delta\varphi_t \\
\partial_t \hat{\varphi}_t + \nabla\cdot(\hat{\varphi}_t f) &= \frac{\sigma_t^2}{2}\Delta\hat{\varphi}_t
\end{aligned}$$

它们分别是 **backward 和 forward Kolmogorov equations**，完全线性且解耦，各自可通过 Feynman-Kac integral representations 独立求解。

## 为什么有效

原始系统中唯一的非线性项是 $\frac{\sigma_t^2}{2}\|\nabla\psi\|^2$。关键技巧是利用 **Laplacian of log**：

$$\Delta\log\varphi = \frac{\Delta\varphi}{\varphi} - \frac{\|\nabla\varphi\|^2}{\varphi^2}$$

设 $\psi = C\log\varphi$ 且 $C=1$，则 $\frac{\sigma^2}{2}\|\nabla\psi\|^2$ 被 $\frac{\sigma^2}{2}\Delta\log\varphi$ 中的 squared gradient 项抵消，quadratic term 完全消失，仅剩线性项。

> [!note] 为什么 $C=1$？
> 若 $\psi = C\log\varphi$，扩散项产生 $\frac{\sigma_t^2}{2}C\Delta\log\varphi = \frac{\sigma_t^2}{2}C(\frac{\Delta\varphi}{\varphi} - \frac{\|\nabla\varphi\|^2}{\varphi^2})$。非线性项 $\frac{\sigma_t^2}{2}C^2\frac{\|\nabla\varphi\|^2}{\varphi^2}$ 与 Laplacian 中的 $\frac{\sigma_t^2}{2}C\frac{\|\nabla\varphi\|^2}{\varphi^2}$ 相消仅当 $C^2 = C$，即 $C = 1$（或平凡解 $C=0$）。

## Density Factorization

$p_t^\star = \varphi_t \hat{\varphi}_t$ 将 optimal marginal 分解为 **forward** 和 **backward** 分量。这些 potentials 满足 Feynman-Kac 表示：

$$\varphi_t(x) = \int Q_{T|t}(y|x)\,\varphi_T(y)\,dy$$

$$\hat{\varphi}_t(x) = \int Q_{t|0}(x|y)\,\hat{\varphi}_0(y)\,dy$$

其中 $Q$ 是 reference process 的 transition density。边界条件给出端点分布的分解：

$$\pi_0 = \varphi_0 \hat{\varphi}_0, \qquad \pi_T = \varphi_T \hat{\varphi}_T$$

## Optimal Control

从变换后的 potentials 恢复最优控制：

$$u^\star = \sigma_t\nabla\log\varphi_t$$

这是 forward Schrödinger potential 的 **score function**。等价地，$u^\star = -\sigma_t\nabla V_t$，其中 $V_t = -\log\varphi_t$ 是 [[stochastic-optimal-control-sb|SOC formulation]] 的 value function。

## 与生成模型的联系

Hopf-Cole 线性化是 score matching 和 flow matching 有效的理论依据：optimal drift（score function）可以通过求解线性 PDE 而非非线性 HJB 方程来学习。这是 variational SB formulation 与 [[diffusion-schrodinger-bridge-matching|DSBM]] 等实用训练算法之间的桥梁。

[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]
