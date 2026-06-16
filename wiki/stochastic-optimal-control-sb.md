---
title: "Stochastic Optimal Control for Schrödinger Bridges"
type: concept
tags:
  - stochastic-optimal-control
  - hjb-equation
  - bellman-optimality
  - schrödinger-bridge
created: 2026-06-16
last_updated: 2026-06-16
source_count: 1
confidence: medium
status: active
---

# Stochastic Optimal Control for Schrödinger Bridges

Stochastic optimal control（SOC）将 [[schrodinger-bridge|Schrödinger bridge problem]] 重新表述为学习一个 optimal control drift $u^\star$，该 drift 在最小化 control cost 的同时将 reference SDE 引导至 prescribed marginals 之间[^src-schrodinger-bridges-generative-modeling]。

## SOC Objective

$$\inf_u \mathbb{E}_{\mathbb{P}^u}\!\left[\int_0^T \left(\frac{1}{2}\|u\|^2 + c\right)dt + \Phi(X_T)\right]$$

s.t. $dX_t^u = (f+\sigma_t u)dt + \sigma_t dB_t$，$X_0^u \sim \pi_0$。

## Value Function 和 HJB Equation

**Value function** $V_t(x) = \inf_u J(x,t;u)$ 是 optimal cost-to-go：

$$V_t(x) = \inf_u \mathbb{E}\!\left[\int_t^T \left(\frac{1}{2}\|u\|^2 + c\right)ds + \Phi(X_T) \middle| X_t = x\right]$$

满足 **Hamilton-Jacobi-Bellman (HJB) equation**：

$$\partial_t V_t = -(A_t V_t) + \frac{\sigma_t^2}{2}\|\nabla V_t\|^2 - c,\quad V_T = \Phi$$

其中 $A_t V_t = \langle f, \nabla V_t\rangle + \frac{\sigma_t^2}{2}\Delta V_t$ 是 uncontrolled generator。

**最优控制**为：

$$u^\star = -\sigma_t\nabla V_t$$

## Bellman's Principle of Optimality

$$\forall t \leq \tau \leq T: V_t(x) = \inf_u \mathbb{E}\!\left[\int_t^\tau \left(\frac{1}{2}\|u\|^2 + c\right)ds + V_\tau(X_\tau^u) \middle| X_t = x\right]$$

取 infinitesimal $\tau \to t$ 即得 HJB equation。

## Feynman-Kac 闭式解

应用 [[hopf-cole-transform|Hopf-Cole transform]] $\varphi_t = e^{-V_t}$ 得到线性 PDE：

$$\partial_t \varphi_t + \langle f, \nabla\varphi_t\rangle + \frac{\sigma_t^2}{2}\Delta\varphi_t - c\varphi_t = 0$$

闭式解：

$$V_t(x) = -\log \mathbb{E}_{\mathbb{Q}}\!\left[\exp\!\left(-\int_t^T c\,ds - \Phi(X_T)\right) \middle| X_t = x\right]$$

## SB-SOC 连接

对 SB，设 $c \equiv 0$ 且 terminal cost $\Phi = \log\frac{\pi_T}{\hat{\varphi}_T}$（使用 Schrödinger potentials）：

$$V_t = -\log\varphi_t, \quad u^\star = \sigma_t\nabla\log\varphi_t$$

这**消除了 initial value function bias**，无需 memoryless reference processes。

## 三种损失族

所有损失均在 $u = u^\star$ 处最小化：

| 损失 | 定义 | 性质 |
|------|------|------|
| **Relative Entropy** | $\mathrm{KL}(\mathbb{P}^u\|\mathbb{P}^\star)$ | 需要 SDE backprop |
| **Cross Entropy** | $\mathrm{KL}(\mathbb{P}^\star\|\mathbb{P}^u)$ | 关于 $\mathbb{P}^u$ 凸，off-policy |
| **Log-Variance** | $\mathrm{Var}[\log\frac{d\mathbb{P}^\star}{d\mathbb{P}^u}]$ | 在 $u=u^\star$ 处为零 |

### Cross-Entropy Loss（path-integral 形式）

$$L_{\mathrm{CE}}(u) = \mathbb{E}_{\mathbb{P}^v}\!\left[e^{-g(X_T^v)}\!\left(\frac{1}{2}\int_0^T\|u\|^2 dt - \int_0^T (u\cdot v)dt - \int_0^T u^\top dB_t^v - g(X_T^v)\right)\right]$$

## Optimal Path Measure

$$\mathbb{P}^\star = \frac{1}{Z}\mathbb{Q}\,\varphi_T\frac{\hat{\varphi}_0}{\pi_0}$$

[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]
