---
title: "Generalized Schrödinger Bridge"
type: concept
tags:
  - schrödinger-bridge
  - mean-field
  - interaction
  - mckean-vlasov
created: 2026-06-16
last_updated: 2026-06-16
source_count: 1
confidence: high
status: active
---

# Generalized Schrödinger Bridge

引入 mean-field interaction，dynamics 取决于整体 population distribution $p_t$。[^src-schrodinger-bridges-generative-modeling]

SDE with interaction term：

$$dX_t = (f(X_t,t) + \sigma_t u(X_t,t) + I(X_t,p_t,t))dt + \sigma_t dB_t$$

## Generalized SB Objective

保留了 interaction cost：[^src-schrodinger-bridges-generative-modeling]

$$\inf_u \mathbb{E}_{\mathbb{P}^u}\!\left[\int_0^T\!\left(\frac{1}{2}\|u\|^2 + c(X_t,t)\right)dt + \Phi(X_T)\right]$$

s.t. $p_0=\pi_0, p_T=\pi_T$，其中 $c$ 可能依赖 population density。

最优控制 $u^\star = \sigma_t\nabla\log\varphi_t$，$\varphi_t$ 因 interaction term 而满足 nonlinear PDE。当 $I \equiv 0$ 时恢复 classical SB。[^src-schrodinger-bridges-generative-modeling]

## 应用

细胞群动力学（cell-cell interactions）、multi-agent systems、crowd modeling。[^src-schrodinger-bridges-generative-modeling]

## 参见

- [[schrodinger-bridge]] — classical SB 理论
- [[stochastic-optimal-control-sb]] — SB 的随机最优控制视角

[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]