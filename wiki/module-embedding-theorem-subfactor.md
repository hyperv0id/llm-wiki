---
title: "Module Embedding Theorem (Subfactor)"
type: concept
tags:
  - operator-algebra
  - subfactor-theory
  - planar-algebra
  - module-category
created: 2026-07-16
last_updated: 2026-07-20
source_count: 1
confidence: medium
status: active
---

# Module Embedding Theorem (Subfactor)

模嵌入定理（Module Embedding Theorem）是子因子平面代数理论的核心结构定理，由 Coles, Huston, Penneys & Srinivas (2018) 在 towers of algebras 框架下证明[^src-module-embedding-theorem-towers-algebras]。

## 陈述

**定理**：设 $\mathcal{Q}_{\bullet}$ 为有限深子因子平面代数，$\mathcal{M}_{\bullet}$ 为其连通右平面模，$\Gamma$ 为 $\mathcal{M}_{\bullet}$ 关于生成子的融合图。则存在平面 †-代数嵌入：

$$\mathcal{Q}_{\bullet} \hookrightarrow \text{GPA}(\Gamma)_{\bullet}$$

其中 $\text{GPA}(\Gamma)_{\bullet}$ 是 $\Gamma$ 的二部图平面代数。

## 三条进路

该定理连接了三条独立进路[^src-module-embedding-theorem-towers-algebras]：

| 进路 | 对象 | 模概念 |
|------|------|--------|
| 张量范畴 | unitary 2×2 multitensor category $\mathcal{C}$ | pivotal right $\mathcal{C}$-module C\* category |
| 平面代数 | subfactor planar algebra $\mathcal{P}_{\bullet}$ | connected right planar module |
| 代数塔 | Markov tower（模 $d$ 的有限维 vNA 塔 + Jones 投影） | 主图 = 融合图 |

**定理 A** 建立前两条进路的等价；Markov tower 作为统一的代数语言连接所有三者。

## 与原 Jones-Penneys 定理的关系

取 $\mathcal{M} = \mathcal{C}_{00} \oplus \mathcal{C}_{10}$（对应无阴影空图）时恢复 Jones-Penneys (2011) 原定理——嵌入到主图的图平面代数中。取 $\mathcal{M} = \mathcal{C}_{10} \oplus \mathcal{C}_{11}$ 和另一单纯对象 $m \in \mathcal{C}_{10}$ 时得到对偶主图嵌入。

## 推论 B：Temperley-Lieb-Jones 模分类

半单 pivotal C\* $\text{TLJ}(d)$-模范畴的等价类，与带 quantum dimension function（满足 Frobenius-Perron 特征值方程 $d \cdot \dim(v) = \sum_{w \sim v} \dim(w)$）的 pointed 连通二部图一一对应[^src-module-embedding-theorem-towers-algebras]。

## 意义

该定理解释了为何某些子因子构造（如 extended Haagerup）可以通过嵌入到与主图完全不同的二部图中实现——这些图正对应不同的循环模选择。

[^src-module-embedding-theorem-towers-algebras]: [[source-module-embedding-theorem-towers-algebras]]
