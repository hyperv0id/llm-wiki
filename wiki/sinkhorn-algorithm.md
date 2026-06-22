---
title: "Sinkhorn Algorithm"
type: technique
tags:
  - optimal-transport
  - iterative-algorithm
  - entropy-regularization
  - schrödinger-bridge
  - doubly-stochastic
created: 2026-06-16
last_updated: 2026-06-22
source_count: 2
confidence: high
status: active
---

# Sinkhorn's Algorithm

Sinkhorn's algorithm（也称 Iterative Proportional Fitting, IPF）是求解 **static Schrödinger bridge problem**（等价于 entropic optimal transport problem）的经典方法。它交替优化两个 dual Schrödinger potentials $(\varphi, \hat{\varphi})$ 以强制 marginal constraints $\pi_0$ 和 $\pi_T$[^src-schrodinger-bridges-generative-modeling]。

## Schrödinger System

给定 cost $c(x,y)$ 和 marginals $\pi_0, \pi_T$，potentials 满足两个耦合方程：

$$\begin{aligned}
\varphi(x) &= -\log\int e^{\hat{\varphi}(y)-c(x,y)}\pi_T(dy) \\
\hat{\varphi}(y) &= -\log\int e^{\varphi(x)-c(x,y)}\pi_0(dx)
\end{aligned}$$

## 算法

初始化 $\varphi_0 := 0$，然后交替更新 $n = 0, 1, \ldots$：

1. **更新** $\hat{\varphi}_n$：用固定的 $\varphi_n$ 求解第二个方程
2. **更新** $\varphi_{n+1}$：用固定的 $\hat{\varphi}_n$ 求解第一个方程

等价地，每一步是对 dual objective 的 coordinate ascent：

$$G(\varphi, \hat{\varphi}) = \int \varphi d\pi_0 + \int \hat{\varphi} d\pi_T - \int e^{\varphi \oplus \hat{\varphi}} dq + 1$$

## 核心性质

### KL Step Identity

每次 Sinkhorn iteration 等于 potentials 的差：

$$\begin{aligned}
\mathrm{KL}(\pi^{2n}\|\pi^{2n-1}) &= \int (\hat{\varphi}_n - \hat{\varphi}_{n-1})\pi_T \\
\mathrm{KL}(\pi^{2n+1}\|\pi^{2n}) &= \int (\varphi_{n+1} - \varphi_n)\pi_0
\end{aligned}$$

### Telescoping Sum

Total dual potential 表达为 accumulated KL：

$$\pi_T(\hat{\varphi}_n) = \sum_{k=0}^n \mathrm{KL}(\pi^{(2k)}\|\pi^{(2k-1)}), \quad \pi_0(\varphi_n) = \sum_{k=0}^{n-1} \mathrm{KL}(\pi^{(2k+1)}\|\pi^{(2k)})$$

## 收敛性

### Marginal Convergence

$$\mathrm{KL}(\pi^{(n)}\|\pi^\star) = \mathrm{KL}(\pi^{(0)}\|q) - \sum_{k=0}^n \mathrm{KL}(\pi^{(k)}\|\pi^{(k-1)})$$

当 $n \to \infty$ 时，$\mathrm{KL}(\pi^{(n)}\|\pi^\star) \to 0$。

### Strong Convergence

在 exponential integrability condition $\exists r>1: \int e^{r c(x,y)} d(\pi_0 \otimes \pi_T) < \infty$ 下，iterates 收敛到 true Schrödinger potentials：$\varphi_n \to \varphi^\star, \hat{\varphi}_n \to \hat{\varphi}^\star$。

## 与现代生成建模的关系

Sinkhorn's algorithm 的交替优化结构在现代生成建模中反复出现：

- [[diffusion-schrodinger-bridge-matching|IMF/DDSBM]] 交替 Markovian 和 reciprocal projections
- [[adjoint-matching]] 交替 forward 和 backward half-bridge 优化
- Likelihood training 交替 forward 和 backward potential 学习

与 [[iterative-markovian-fitting]] 的关系：两者都是交替投影，Sinkhorn 作用于 coupling 空间，IMF 作用于 path measure 空间。

## 双随机矩阵缩放（Sinkhorn-Knopp 矩阵缩放）

除了求解 Schrödinger bridge，Sinkhorn-Knopp 迭代也常用于将非负矩阵投影到 Birkhoff 多面体上，即生成行和、列和均为 1 的双随机矩阵[^src-mhc-manifold-constrained-hyper-connections]。给定矩阵 $\tilde{M}$，算法先通过指数化得到正矩阵 $M^{(0)} = \exp(\tilde{M})$，然后交替进行行归一化与列归一化：

$$M^{(t)} = T_r(T_c(M^{(t-1)}))$$

其中 $T_r, T_c$ 分别表示逐行、逐列 rescale 到和为 1。当 $t \to \infty$ 时，$M^{(t)}$ 收敛到一个双随机矩阵[^src-mhc-manifold-constrained-hyper-connections]。

在 [[manifold-constrained-hyper-connections|mHC]] 中，这一操作被用来约束残差流之间的混合矩阵 $H_l^{res}$，使其保持恒等映射的稳定性质[^src-mhc-manifold-constrained-hyper-connections]。

[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]
[^src-mhc-manifold-constrained-hyper-connections]: [[source-mhc-manifold-constrained-hyper-connections]]
