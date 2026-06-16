---
title: "Entropic Optimal Transport"
type: concept
tags:
  - optimal-transport
  - entropy
  - kl-divergence
  - schrödinger-bridge
created: 2026-06-16
last_updated: 2026-06-16
source_count: 1
confidence: medium
status: active
---

# Entropic Optimal Transport

Entropic optimal transport（EOT）是带有 entropy regularization 的 classical optimal mass transport 扩展。通过添加 entropy penalty（KL divergence from a reference coupling），EOT 产生**唯一、平滑、随机的 coupling**，与 classical OMT 可能非唯一、deterministic 的解形成对比[^src-schrodinger-bridges-generative-modeling]。

## 定义

给定 marginals $\pi_0, \pi_T$、cost function $c(x,y)$、reference coupling $q$ 和正则化参数 $\alpha > 0$：

$$\min_{\pi \in \Pi(\pi_0,\pi_T)} \int c\,d\pi + \alpha\,\mathrm{KL}(\pi\|q)$$

其中 $\Pi(\pi_0,\pi_T)$ 是具有指定 marginals 的 coupling 集合。

## 化为 KL Projection

通过 tilting reference measure 为 $\tilde{q} \propto e^{-c/\alpha}(\pi_0\otimes\pi_T)$，EOT 问题变为纯 KL projection：

$$\min_{\pi \in \Pi(\pi_0,\pi_T)} \mathrm{KL}(\pi\|\tilde{q})$$

这揭示了 EOT 是将 tilted reference 投影到 marginal constraint set 上的 **entropy projection**，正是 [[schrodinger-bridge|static Schrödinger bridge problem]] 的变分结构。

## 三个动机

1. **Stochastic coupling**：与 OMT 的 sparse/deterministic maps 不同，entropy regularization 产生在 $\mathcal{X} \times \mathcal{Y}$ 上平滑扩散的概率质量
2. **Strict convexity**：entropy 函数 $\pi \mapsto \int \pi \log \pi$ 是凸的，保证**唯一 minimizer**
3. **OMT 的推广**：$\alpha \to 0$ 时恢复 classical OMT；$\alpha \to \infty$ 时解趋近 reference coupling

## 关键性质

### KL Chain Rule

对于 joint measures，entropy 分解为：

$$\mathrm{KL}(\pi_{\mathcal{X},\mathcal{Y}}\|\pi'_{\mathcal{X},\mathcal{Y}}) = \mathrm{KL}(\pi_{\mathcal{X}}\|\pi'_{\mathcal{X}}) + \mathbb{E}_{x\sim\pi_{\mathcal{X}}}[\mathrm{KL}(\pi_{\mathcal{Y}|\mathcal{X}}\|\pi'_{\mathcal{Y}|\mathcal{X}})]$$

### Data Processing Inequality

对两个分布施加相同的 Markov kernel 不会增加其 divergence：

$$\mathrm{KL}(\tilde{p}\|\tilde{q}) \leq \mathrm{KL}(p\|q)$$

## 与 Schrödinger Bridge 的关系

Static SB 问题在 $q = e^{-c}(\pi_0\otimes\pi_T)$ 时**等价于** EOT 问题。最优 coupling 通过 Schrödinger potentials $(\varphi, \hat{\varphi})$ 分解，满足 Schrödinger system：

$$\begin{aligned}
\varphi(x) &= -\log \int e^{\hat{\varphi}(y)-c(x,y)}\pi_T(dy) \\
\hat{\varphi}(y) &= -\log \int e^{\varphi(x)-c(x,y)}\pi_0(dx)
\end{aligned}$$

由 [[sinkhorn-algorithm]] 高效求解。

## Dynamic 推广

提升到 path space 得到 [[schrodinger-bridge|dynamic Schrödinger bridge]]，其中 KL divergence 是受控 SDE 诱导的 path measures 之间的：

$$\mathrm{KL}(\mathbb{P}^{\tilde{u}}\|\mathbb{P}^u) = \mathbb{E}_{\mathbb{P}^{\tilde{u}}}\!\left[\frac{1}{2}\int_0^T \|(\tilde{u}-u)\|^2 dt\right]$$

[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]
