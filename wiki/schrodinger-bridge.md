---
title: "Schrödinger Bridge"
type: concept
tags:
  - schrödinger-bridge
  - optimal-transport
  - generative-modeling
  - entropy-regularization
  - stochastic-processes
created: 2026-06-16
last_updated: 2026-07-18
source_count: 2
confidence: medium
status: active
---

# Schrödinger Bridge

Schrödinger bridge（SB）是一个数学框架，用于确定将一个概率分布变换为另一个的**最似然随机演化过程**，以给定的 reference stochastic process 为基准。它将 diffusion models、score-based generative modeling 和 flow matching 统一在同一个变分原理之下：**在 marginal constraints 下，最小化相对于 reference path measure 的 KL divergence**[^src-schrodinger-bridges-generative-modeling]。

## 核心原理

给定初始分布与终端分布 $\pi_0, \pi_T$ 以及一个 reference stochastic process $\mathbb{Q}$，SB problem 寻求：

$$\mathbb{P}^\star = \arg\min_{\mathbb{P}} \{\mathrm{KL}(\mathbb{P}\|\mathbb{Q}) : p_0 = \pi_0, p_T = \pi_T\}$$

这就是 path space 中的 **entropy-regularized optimal transport**。

在 controlled SDE 下，KL divergence 具有显式表达：

$$\mathrm{KL}(\mathbb{P}^u\|\mathbb{Q}) = \mathbb{E}_{\mathbb{P}^u}\left[\frac{1}{2}\int_0^T \|u(X_t,t)\|^2 dt\right]$$

## 两种形式

### Static SB

作用于 couplings $\pi_{0,T} \in \Pi(\pi_0, \pi_T)$ 上：

$$\pi_{0,T}^\star = \arg\min_{\pi_{0,T} \in \Pi(\pi_0,\pi_T)} \mathrm{KL}(\pi_{0,T}\|q)$$

解通过 **Schrödinger potentials** $(\varphi, \hat{\varphi})$ 分解：

$$\pi_{0,T}^\star = e^{\varphi(x)+\hat{\varphi}(y)-c(x,y)} \pi_0 \otimes \pi_T$$

由 [[sinkhorn-algorithm|Sinkhorn 算法]]（iterative proportional fitting）求解。

### Dynamic SB

作用于由 controlled SDE 诱导的 path measures $\mathbb{P} \in \mathcal{P}(C([0,T];\mathbb{R}^d))$ 上：

$$dX_t = (f(X_t,t) + \sigma_t u(X_t,t))dt + \sigma_t dB_t$$

最优控制取形式 $u^\star = \sigma_t \nabla \log \varphi_t$，其中 $(\varphi_t, \hat{\varphi}_t)$ 满足 **Hopf-Cole 线性化 PDE**：

$$\begin{aligned}
\partial_t \varphi_t + \langle\nabla\varphi_t, f\rangle &= -\frac{\sigma_t^2}{2}\Delta\varphi_t \\
\partial_t \hat{\varphi}_t + \nabla\cdot(\hat{\varphi}_t f) &= \frac{\sigma_t^2}{2}\Delta\hat{\varphi}_t
\end{aligned}$$

且 marginal 满足 $p_t^\star = \varphi_t \hat{\varphi}_t$。

## 生成模型的统一框架

SB 将若干范式统一在一个框架之下：

| 框架 | SB 解释 |
|------|--------|
| Diffusion models | 固定前向过程（通常为 OU/VE/VP），学习 reverse-time score |
| Score-based models | Reference = Brownian motion，学习 $u^\star = \sigma_t \nabla \log \varphi_t$ |
| Flow matching | Reference = $\sigma_t \to 0$ 极限，恢复 deterministic OT |
| Stochastic optimal control | SB 视作以 $\log(\hat{\varphi}_T/\pi_T)$ 为 terminal cost 的 SOC |

> [!note] 统一原理
> 以上所有框架都归结为同一个变分问题：寻找从 $\pi_0$ 到 $\pi_T$ 的 reference process 的最小 KL perturbation。区别仅在于 $\sigma_t$ 的选择和 reference 的构造方式。

## 构造方法

构建 SB 的六种互补方法：

1. **Mixture of conditional bridges** — 将 dynamic SB 分解为 static SB + 端点条件下的 reference bridges
2. **Time reversal** — 带 score correction 的 reverse SDE
3. **Forward-backward SDEs** — 耦合 FBSDE 系统
4. **Doob's h-transform** — 概率倾斜 $h(x,t) = \mathbb{E}_Q[\varphi_T(X_T)|X_t=x]$
5. **Markovian/reciprocal projections** — Iterative Markovian Fitting (IMF)
6. **Stochastic interpolants** — $x_t = I^\star(x_0,x_T,t) + \gamma(t)z$

详见 [[building-schrodinger-bridges]]。

## 关键变体

- [[gaussian-schrodinger-bridge]] — Gaussian marginals 的闭式解
- [[generalized-schrodinger-bridge]] — 含 mean-field interactions
- [[multi-marginal-schrodinger-bridge]] — 多个中间约束
- [[unbalanced-schrodinger-bridge]] — 允许质量生成/消失
- [[branched-schrodinger-bridge]] — 多模态终端分布
- [[fractional-schrodinger-bridge]] — 通过 fBM 引入长程依赖

## 应用案例

### GMF：多模态融合的几何可靠性

[[gmf|GMF]] (Geometry-based Multimodal Fusion) 将 Schrödinger bridge 应用于多模态融合中的可靠性评估[^src-gmf]。通过单步 Rectified Flow（DSB 近似）在潜在空间中估计模态内和模态间传输代价，作为与分类器输出解耦的外在可靠性信号。跨模态传输代价上的[[geometric-barrier-principle|几何屏障]]（$\geq (\delta-2\epsilon)^2$）能指数级抑制语义冲突模态的融合权重，打破传统统计方法的[[circular-dependency-in-multimodal-fusion|循环依赖]]。详见 [[transport-based-reliability-assessment|基于传输的可靠性评估]]。

[^src-gmf]: [[source-gmf]]

[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]