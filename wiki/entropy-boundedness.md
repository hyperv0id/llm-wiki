---
title: "Entropy Boundedness"
type: technique
tags:
  - position-encoding
  - extrapolation
  - theory
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

**熵有界性 (Entropy Boundedness)** 是无限上下文外推的关键数学条件之一，确保注意力分布的 Shannon 熵有限[^src-vetcha-2026-towards-infinite-length-extrapolation]。

## 定义

设归一化注意力权重 $p(n) = e^{A(n)} / \sum_{k \in \mathbb{Z}} e^{A(k)}$，若：

$$H = -\sum_{n \in \mathbb{Z}} p(n) \log p(n) < \infty$$

则该位置编码方法满足熵有界性[^src-vetcha-2026-towards-infinite-length-extrapolation]。

## 意义

- 防止注意力分布过于平坦（高熵）或过于集中（低熵）
- 平衡局部关注和全局建模能力
- 是稳定学习和推理的必要条件

## 定理 3.5：等价性

Vetcha 2026 证明了收敛归一化与熵有界性的等价性[^src-vetcha-2026-towards-infinite-length-extrapolation]：

1. **若 $Z < \infty$，则 $H < \infty$**
2. **若对所有 $L$，截断分布 $p_L(n) = e^{A(n)} / \sum_{k=-L}^{L} e^{A(k)}$ 的熵一致有界，则 $Z < \infty$**

这意味着：满足收敛归一化 ⇒ 自动满足熵有界性。

---

[^src-vetcha-2026-towards-infinite-length-extrapolation]: [[source-vetcha-2026-towards-infinite-length-extrapolation]]