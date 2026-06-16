---
title: "Unbalanced Schrödinger Bridge"
type: concept
tags:
  - schrödinger-bridge
  - unbalanced-transport
  - wasserstein-fisher-rao
  - population-dynamics
created: 2026-06-16
last_updated: 2026-06-16
source_count: 1
confidence: high
status: active
---

# Unbalanced Schrödinger Bridge

允许质量产生/消灭的 SB 变体。Fokker-Planck 加了 growth term $g$：[^src-schrodinger-bridges-generative-modeling]

$$\partial_t p_t = -\nabla\cdot(p_t(f+\sigma_t u)) + \frac{\sigma_t^2}{2}\Delta p_t + g p_t$$

## Unbalanced SB Objective

$$\inf_{p_t,u,g}\int_0^T\int_{\mathbb{R}^d}\left[\frac{1}{2}\|u\|^2 + \alpha\Psi(g)\right]p_t dx dt$$

s.t. $p_{t_k} = \pi_{t_k}$ at observation times。$\Psi(g)$ 罚质量变化，如 KL 形式 $\Psi(g)=g\log g - g + 1$。[^src-schrodinger-bridges-generative-modeling]

$\alpha \to \infty$ 极限恢复 balanced SB。$\alpha \to 0$ 允许自由增减。[^src-schrodinger-bridges-generative-modeling]

## Wasserstein-Fisher-Rao Distance

Unbalanced OT 的度量结构，在 $L^2$-Wasserstein（纯传输）和 Fisher-Rao（纯质量变化）之间插值。[^src-schrodinger-bridges-generative-modeling]

## 应用

细胞增殖/死亡建模、population dynamics with changing total mass。[^src-schrodinger-bridges-generative-modeling]

## 参见

- [[schrodinger-bridge]] — balanced SB 理论
- [[multi-marginal-schrodinger-bridge]] — 多时间点扩展

[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]