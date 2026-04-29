---
title: "Convergent Normalization"
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

**收敛归一化 (Convergent Normalization)** 是无限上下文外推的关键数学条件之一，确保注意力分布的 softmax 分母在无限序列长度下收敛[^src-vetcha-2026-towards-infinite-length-extrapolation]。

## 定义

设 $Z = \sum_{n=0}^{L} e^{A(n)}$ 为 softmax 归一化常数，若：

$$\lim_{L \to \infty} Z < \infty$$

则该位置编码方法满足收敛归一化[^src-vetcha-2026-towards-infinite-length-extrapolation]。

## 意义

- 确保注意力分布在无限长度下仍有定义
- 防止远距离 token 的注意力权重之和发散导致分布崩溃
- 是熵有界性的必要条件（定理 3.5）[^src-vetcha-2026-towards-infinite-length-extrapolation]

## 方法对比

| 方法 | 收敛归一化 |
|------|-----------|
| RoPE | ❌ — 无衰减因子，$e^{A(n)}$ 不随 $\|n\|$ 衰减 |
| ALiBi | ✅ — $e^{-m\|n\|}$ 指数衰减 |
| APE | ✅ — $e^{-\delta\|n\| - \gamma\sqrt{\|n\|}}$ 指数衰减 |

---

[^src-vetcha-2026-towards-infinite-length-extrapolation]: [[source-vetcha-2026-towards-infinite-length-extrapolation]]