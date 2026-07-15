---
title: "Subfactor Planar Algebra"
type: entity
tags:
  - operator-algebra
  - subfactor-theory
  - planar-algebra
created: 2026-07-16
last_updated: 2026-07-20
source_count: 1
confidence: medium
status: active
---

# Subfactor Planar Algebra

子因子平面代数（subfactor planar algebra）是 II₁ 子因子的标准不变量的一种公理化，由 Vaughan Jones 于 1999 年引入[^src-module-embedding-theorem-towers-algebras]。它统一了此前 Popa 的 λ-lattice 进路和 Jones 的 index 理论。

## 定义

子因子平面代数是 shaded planar operad 上的代数，由一族有限维 C-向量空间 ${P_{n,\pm}}$ 构成，满足：

1. **有限维**：$\dim(P_{n,\pm}) < \infty$
2. **可评估/连通**：$\dim(P_{0,\pm}) = 1$
3. **正定性**：每个 $P_{n,\pm}$ 上由 tangle 给出的内积正定
4. **球面性**：左右迹一致

闭合可缩圈替换为乘法标量 $d > 0$（loop parameter），由 Jones 指数刚性定理，$d \in \{2\cos(\pi/k) \mid k \geq 3\} \cup [2, \infty)$。

## 与 unitary 2×2 multitensor category 的等价

子因子平面代数等价于带生成子 $X \in \mathcal{C}_{01}$ 的 unitary 2×2 multitensor category $\mathcal{C}$。从 $\mathcal{P}_{\bullet}$ 构造 $\mathcal{C}$ 通过投影范畴（projection category）；反之通过 $\mathcal{C}(X^{\text{alt} \otimes n})$ 定义 $P_{n,\pm}$，以 pivotal 张量范畴的图示演算定义 operad 作用[^src-module-embedding-theorem-towers-algebras]。

## 模嵌入定理

有限深子因子平面代数可嵌入其循环模的融合图的二部图平面代数中，推广了 Jones-Penneys (2011) 仅考虑主图的情形（参见 [[module-embedding-theorem-subfactor|模嵌入定理]]）[^src-module-embedding-theorem-towers-algebras] 。

[^src-module-embedding-theorem-towers-algebras]: [[source-module-embedding-theorem-towers-algebras]]
