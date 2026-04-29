---
title: "Generalized Positional Encoding (GPE) Framework"
type: concept
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

**Generalized Positional Encoding (GPE) 框架**是 Vetcha 2026 提出的统一理论框架，将位置编码对注意力分数的修改分解为乘法变换和加性偏置两部分[^src-vetcha-2026-towards-infinite-length-extrapolation]。

## 形式化定义

给定查询向量 $q_i$ 和键向量 $k_j$，相对位置 $n = i - j$，GPE 修改原始注意力分数为：

$$A(n) = f(n) \cdot q_i^\top W(n) k_j + b(n)$$

其中[^src-vetcha-2026-towards-infinite-length-extrapolation]：

- $W(n) \in \mathbb{R}^{d \times d}$：位置依赖的变换矩阵（如旋转矩阵）
- $f(n) : \mathbb{Z} \to \mathbb{R}^+$：增益/衰减函数
- $b(n) : \mathbb{Z} \to \mathbb{R}$：加性偏置函数

## 现有方法的统一表示

| 方法 | $W(n)$ | $f(n)$ | $b(n)$ |
|------|--------|--------|--------|
| RoPE | $R(n)$ (旋转矩阵) | 1 | 0 |
| ALiBi | $I$ | 1 | $-m\|n\|$ |
| APE | $R_{\alpha(n)}(n)$ | $1/(1+\lambda\|n\|)$ | $-\delta\|n\| - \beta\log(1+\|n\|) - \gamma\sqrt{\|n\|}$ |

## 框架意义

该框架揭示了现有方法的本质差异和内在局限：

1. **RoPE** 通过旋转矩阵实现相对位置编码，但缺少衰减机制，导致归一化常数发散[^src-vetcha-2026-towards-infinite-length-extrapolation]
2. **ALiBi** 通过线性偏置实现衰减，但完全丢失了远距离相关性[^src-vetcha-2026-towards-infinite-length-extrapolation]
3. **APE** 结合两者优点，通过次线性衰减保留长程依赖[^src-vetcha-2026-towards-infinite-length-extrapolation]

---

[^src-vetcha-2026-towards-infinite-length-extrapolation]: [[source-vetcha-2026-towards-infinite-length-extrapolation]]