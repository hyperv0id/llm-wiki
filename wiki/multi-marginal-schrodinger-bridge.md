---
title: "Multi-Marginal Schrödinger Bridge"
type: concept
tags:
  - schrödinger-bridge
  - multi-marginal
  - sinkhorn
  - trajectory-inference
created: 2026-06-16
last_updated: 2026-06-16
source_count: 1
confidence: high
status: active
---

# Multi-Marginal Schrödinger Bridge

将 SB 扩展到 $K \geq 2$ 个中间时间点的 marginal constraints $\pi_{t_k}$ at $0=t_0<\cdots<t_K=T$。[^src-schrodinger-bridges-generative-modeling]

$$\mathbb{P}^\star = \arg\min_{\mathbb{P}} \{\mathrm{KL}(\mathbb{P}\|\mathbb{Q}) : p_{t_k} = \pi_{t_k}, \forall k\}$$

最优 measure 分解为 Markov chain：[^src-schrodinger-bridges-generative-modeling]

$$\mathbb{P}^\star(X_0,\ldots,X_{t_K}) \propto \mathbb{Q}(X_0,\ldots,X_{t_K})\prod_{k=0}^K e^{-\Phi_k(X_{t_k})}$$

## Multi-Marginal Sinkhorn

可通过 multi-marginal Sinkhorn 求解（对每个 marginal constraint 交替投影）。Dynamic 化为分段受控 SDE：[^src-schrodinger-bridges-generative-modeling]

$$dX_t = (f + \sigma_t u_k^\star)dt + \sigma_t dB_t, \quad t\in[t_{k-1},t_k]$$

## 应用

- 单细胞动力学多时间点 snapshots
- Trajectory inference from population time-series

当有中间快照但无个体轨迹时，multi-marginal 形式是自然选择。[^src-schrodinger-bridges-generative-modeling]

## 参见

- [[schrodinger-bridge]] — 双 marginal SB 理论
- [[sinkhorn-algorithm]] — 最优传输的迭代缩放算法
- [[unbalanced-schrodinger-bridge]] — 允许质量变化的扩展

[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]