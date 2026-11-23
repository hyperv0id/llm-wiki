---
title: "Stochastic Optimal Control for Schrödinger Bridges"
type: concept
tags:
  - stochastic-optimal-control
  - hjb-equation
  - bellman-optimality
  - schrödinger-bridge
created: 2026-06-16
last_updated: 2026-11-22
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

其中 $A_t V_t = \langle f, \nabla V_t\rangle + \frac{\sigma_t^2}{2}\Delta V_t$ 是 uncontrolled generator。$\frac{\sigma_t^2}{2}\|\nabla V_t\|^2$ 一项让 HJB 成为非线性 PDE，这是该表述的核心难点，也是需要 [[hopf-cole-transform|Hopf-Cole transform]] 的原因。

**最优控制**为：

$$u^\star = -\sigma_t\nabla V_t$$

它是 Hamiltonian $H(x,u,\nabla V)=\langle u,\nabla V\rangle + \frac{1}{2}\|u\|^2$ 的逐点极小点。

## Bellman's Principle of Optimality

$$\forall t \leq \tau \leq T: V_t(x) = \inf_u \mathbb{E}\!\left[\int_t^\tau \left(\frac{1}{2}\|u\|^2 + c\right)ds + V_\tau(X_\tau^u) \middle| X_t = x\right]$$

取 infinitesimal $\tau \to t$ 即得 HJB equation。

## Feynman-Kac 闭式解

应用 [[hopf-cole-transform|Hopf-Cole transform]] $\varphi_t = e^{-V_t}$ 得到线性 PDE：

$$\partial_t \varphi_t + \langle f, \nabla\varphi_t\rangle + \frac{\sigma_t^2}{2}\Delta\varphi_t - c\varphi_t = 0,\quad \varphi_T = e^{-\Phi}$$

当 $c \equiv 0$ 时它退化为 reference process 的前向 Kolmogorov 方程。

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

### Relative Entropy Loss

$$L_{\mathrm{RE}}(u) = \mathbb{E}_{\mathbb{P}^u}\!\left[\frac{1}{2}\int_0^T \|u\|^2\,dt + \int_0^T c\,dt + \Phi(X_T^u)\right]$$

由 SB 的 KL 目标经 Girsanov 定理直接得到，是 $\mathrm{KL}(\mathbb{P}^u\|\mathbb{P}^\star)$ 的无偏估计；rollout 来自当前策略 $\mathbb{P}^u$，训练早期即可用，代价是需要对 SDE 反向传播。

### Cross-Entropy Loss（path-integral 形式）

$$L_{\mathrm{CE}}(u) = \mathbb{E}_{\mathbb{P}^v}\!\left[e^{-g(X_T^v)}\!\left(\frac{1}{2}\int_0^T\|u\|^2 dt - \int_0^T (u\cdot v)dt - \int_0^T u^\top dB_t^v - g(X_T^v)\right)\right]$$

CE 关于 $\mathbb{P}^u$ 凸，但需要来自最优桥 $\mathbb{P}^\star$ 的样本，$\mathbb{P}^\star$ 尚未得到时（训练早期）不实用；RE 用 $\mathbb{P}^u$ 的 rollout，更适合这一阶段。这条凸性与 off-policy 的取舍是选择损失族的主要依据[^src-schrodinger-bridges-generative-modeling]。

## Optimal Path Measure

$$\mathbb{P}^\star = \frac{1}{Z}\mathbb{Q}\,\varphi_T\frac{\hat{\varphi}_0}{\pi_0}$$

## 相关页面

- [[schrodinger-bridge]] — SOC 所重述的路径空间问题
- [[hopf-cole-transform]] — 线性化 HJB 的非线性项
- [[doob-h-transform]] — 通过概率倾斜构造同一控制的替代路线
- [[diffusion-schrodinger-bridge-matching]] — 基于 SOC 目标的实用算法

[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]
