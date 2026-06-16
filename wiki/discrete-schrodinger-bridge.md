---
title: "Discrete Schrödinger Bridge"
type: concept
tags:
  - schrödinger-bridge
  - discrete-state
  - ctmc
  - optimal-transport
created: 2026-06-16
last_updated: 2026-06-16
source_count: 1
confidence: high
status: active
---

# Discrete Schrödinger Bridge

Discrete SB 将 SB 推广到有限状态空间 $\mathcal{X} = \{1,\ldots,d\}$，使用 continuous-time Markov chains (CTMCs)。generator $Q_t(x,y)$ 定义跳跃速率[^src-schrodinger-bridges-generative-modeling]：

$$Q_t(x,y) = \lim_{\Delta t \to 0} \frac{1}{\Delta t}(\Pr(X_{t+\Delta t}=y|X_t=x) - \mathbf{1}_{x=y})$$

## Kolmogorov 方程

**Forward equation**（marginal 分布的演化）：

$$\partial_t p_t(x) = \sum_{y\neq x} (Q_t(y,x)p_t(y) - Q_t(x,y)p_t(x))$$

**Backward equation**（test functions 的演化）：

$$-\partial_t \phi_t(x) = \sum_{y\neq x} (\phi_t(y)-\phi_t(x)) Q_t(x,y)$$

## Discrete SB Problem

$$\mathbb{P}^\star = \arg\min_{\mathbb{P}^u} \{\mathrm{KL}(\mathbb{P}^u\|\mathbb{Q}) : p_0 = \pi_0, p_T = \pi_T\}$$

两个 CTMC path measures 之间的 KL divergence 为[^src-schrodinger-bridges-generative-modeling]：

$$\mathrm{KL}(\mathbb{P}^u\|\mathbb{Q}) = \mathbb{E}_{\mathbb{P}^u}\left[\int_0^T \sum_{y\neq X_t} \left(Q_t^u \log\frac{Q_t^u}{Q_t^0} + Q_t^0 - Q_t^u\right)(X_t,y) dt\right]$$

## Path Radon-Nikodym Derivative

$$\log\frac{d\mathbb{P}'}{d\mathbb{P}} = \log\frac{d\pi_0'}{d\pi_0}(X_0) + \sum_{t: X_{t-}\neq X_t} \log\frac{Q_t'(X_{t-},X_t)}{Q_t(X_{t-},X_t)} + \int_0^T \sum_{y\neq X_t} (Q_t-Q_t')(X_t,y)dt$$

其中跳跃项 $\sum_{t: X_{t-}\neq X_t}$ 对路径上所有跳跃时刻求和，是 CTMC 特有的离散贡献。

## Discrete Stochastic Optimal Control

未受控的 generator $A_t$ 作用于函数 $\phi$：

$$(A_t\phi)(x) = \sum_{y\neq x} Q_t(x,y)(\phi(y)-\phi(x))$$

受控 generator 加入 $u_t(x,y)$，约束 $\sum_y u_t(x,y) = 0$，得到受控速率[^src-schrodinger-bridges-generative-modeling]：

$$Q_t^u(x,y) = Q_t(x,y) + u_t(x,y)$$

## Discrete SB-SOC Objective

$$\min_u \mathbb{E}_{\mathbb{P}^u}\left[\frac{1}{2}\int_0^T \sum_y \|u_t(X_t,y)\|^2 dt + \log \frac{\hat{\varphi}_T(X_T)}{\pi_T(X_T)}\right]$$

## Discrete IMF (DDSBM)

Discrete SB 的 Iterative Markovian Fitting 推广了连续状态空间的 IMF 过程，交替进行 Markovian projection 和 reciprocal projection，适用于离散状态空间上的生成建模。

## 参见

- [[schrodinger-bridge]] — 连续状态 SB 理论
- [[iterative-markovian-fitting]] — 基于 Markov chain 的离散生成建模

[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]