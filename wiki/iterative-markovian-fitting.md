---
title: "迭代 Markov 拟合 (IMF)"
type: technique
tags:
  - schrödinger-bridge
  - markovian-projection
  - reciprocal-projection
  - kl-divergence
created: 2026-06-16
last_updated: 2026-06-16
source_count: 1
confidence: medium
status: active
source_count: 1
confidence: high
status: active
---

# 迭代 Markov 拟合 (IMF)

迭代 Markov 拟合（Iterative Markovian Fitting, IMF）是 [[schrodinger-bridge|Schrödinger bridge]] 的核心构造算法：交替应用 Markovian projection 和 reciprocal projection，最终收敛到唯一的最优路径测度 $\mathbb{P}^\star$[^src-schrodinger-bridges-generative-modeling]。

## Markovian Projection

Markovian projection 在 KL 散度意义下，将任意路径测度 $\Pi$ 投影到 Markov 测度空间 $\mathcal{M}$：

$$\mathbb{M} = \mathrm{proj}_{\mathcal{M}}(\Pi) = \arg\min_{\mathbb{M}\in\mathcal{M}} \mathrm{KL}(\Pi\|\mathbb{M})$$

该投影保持所有单一时点 marginal 不变：$\mathbb{M}_t = \Pi_t$。其 SDE 形式为：

$$dX_t = \Bigl[f + \sigma_t^2\,\mathbb{E}_{\Pi_{T\mid t}}\![\nabla\log Q_{T\mid t}(X_T\mid X_t)\mid X_t]\Bigr]dt + \sigma_t dB_t$$

条件期望内的项正是参考过程条件 score 的最优预测。

## Reciprocal Projection

Reciprocal projection 在保持 bridge 结构的同时强制执行端点约束：

$$\Pi = \mathrm{proj}_{\mathcal{R}(\mathbb{Q})}(\mathbb{M}) = \mathbb{M}_{0,T}\,\mathbb{Q}_{\cdot\mid 0,T}$$

其直观含义：以 joint endpoint distribution $\mathbb{M}_{0,T}$ 重新加权，再将两端点之间的路径用参考过程的 conditional bridge $\mathbb{Q}_{\cdot\mid 0,T}$ 填充。

## Pythagorean 恒等式

交替投影的收敛性由以下恒等式保证——对任意 $\Pi \in \mathcal{R}(\mathbb{Q})$ 和 $\mathbb{M} \in \mathcal{M}$：

$$\mathrm{KL}(\Pi\|\mathbb{M}) = \mathrm{KL}(\Pi\|\mathrm{proj}_{\mathcal{M}}(\Pi)) + \mathrm{KL}(\mathrm{proj}_{\mathcal{M}}(\Pi)\|\mathbb{M})$$

这一定理表明 Markovian projection 是 KL 意义下的正交投影，每次投影均严格减小 KL 散度。

## 算法流程

从任意初始 reciprocal 测度出发（通常取 $\Pi_0 = \pi_{0,T}\,\mathbb{Q}_{\cdot\mid 0,T}$，即简单耦合加 reference bridges），交替迭代：

1. **Markovian projection**：$\mathbb{M}_{2n+1} = \mathrm{proj}_{\mathcal{M}}(\Pi_{2n})$
2. **Reciprocal projection**：$\Pi_{2n+1} = \mathbb{M}_{2n+1}\,\mathbb{Q}_{\cdot\mid 0,T}$

交替结构完全对应于离散情形下的 [[sinkhorn-algorithm]]——两者都通过向两个约束集交替投影来求解 entropic optimal transport。

## 收敛性

IMF 具有严格的收敛保证[^src-schrodinger-bridges-generative-modeling]：

- $\mathrm{KL}(\mathbb{P}_n\|\mathbb{P}^\star)$ **单调递减**，$\lim_{n\to\infty}\mathrm{KL}(\mathbb{P}_n\|\mathbb{P}_{n+1}) = 0$。
- 唯一不动点满足 $\mathbb{M}^\star = \Pi^\star = \mathbb{P}^\star$，即 Markovian 和 reciprocal 投影在 SB 最优解处重合。
- 每轮迭代使 KL 散度严格减小，直至收敛。

## 实践实现

IMF 的理论框架有多种参数化实现。[[diffusion-schrodinger-bridge-matching|DSBM]] 是其中最典型的方案——用神经网络参数化 Markov drift，通过 conditional score matching 逼近 Markovian projection，交替学习前向和反向 drift。

## 参见

- [[diffusion-schrodinger-bridge-matching]] — IMF 的参数化实现
- [[sinkhorn-algorithm]] — 离散情形下的交替投影
- [[schrodinger-bridge]] — SB 问题定义
- [[building-schrodinger-bridges]] — 六种构造方法总览

[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]
