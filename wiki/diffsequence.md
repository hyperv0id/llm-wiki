---
title: "Diffsequence (差序列)"
type: concept
tags:
  - ramsey-theory
  - combinatorics
  - additive-combinatorics
created: 2026-07-21
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

# Diffsequence（差序列）

**Diffsequence**（差序列）是 [[ramsey-theory|Ramsey 理论]] 中由 Landman 和 Robertson (2003) 引入的组合概念，研究相邻项之差受限于固定集合的递增整数序列 [^src-new-bounds-on-diffsequences].

## 定义

对于正整数集合 $D$，一个 **$k$ 项 D-diffsequence** 是满足以下条件的正整数序列 $a_1 < a_2 < \cdots < a_k$：

$$a_i - a_{i-1} \in D, \quad \text{for } i = 2, 3, \ldots, k$$

即相邻项之间的**所有 gap**（差值）都必须落在预先指定的集合 $D$ 中 [^src-new-bounds-on-diffsequences]。

当 $D = \{1\}$ 时，D-diffsequence 退化为连续整数。当 $D = \{d\}$ 时即固定公差的等差数列。对于更一般的 $D$，diffsequence 允许 gaps 在不同大小之间切换但仍限于指定集合。

## Accessibility

集合 $D$ 称为 **$r$-accessible**，如果正整数的任意 $r$-染色都包含任意长的单色 D-diffsequence。这是 van der Waerden 定理（等差数列对任意染色存在任意长单色序列）在 diffsequence 上的推广 [^src-new-bounds-on-diffsequences]。

若 $D$ 是 $r$-accessible，定义 **Ramsey 数** $\Delta(D, k; r)$ 为使得 $\{1, 2, \ldots, n\}$ 的任意 $r$-染色包含单色 $k$ 项 D-diffsequence 的最小 $n$。对 2-染色简记为 $\Delta(D, k)$。

## 已知结果

- $D = \{2^i\}$（2 的幂次）：2-accessible，$\Delta(D, k)$ 介于 $2^{\Theta(\sqrt{k})}$ 和 $2^{O(k)}$ 之间 [^src-new-bounds-on-diffsequences]
- $D = \{k!\}$（阶乘）：不是 2-accessible [^src-new-bounds-on-diffsequences]
- Fibonacci 数列：2-accessible 但非 4-accessible [^src-new-bounds-on-diffsequences]
- 素数集合：非 3-accessible，2-accessibility 仍然 open [^src-new-bounds-on-diffsequences]

[^src-new-bounds-on-diffsequences]: [[source-new-bounds-on-diffsequences]]
