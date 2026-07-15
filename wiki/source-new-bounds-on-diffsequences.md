---
title: "New Bounds on Diffsequences"
type: source-summary
tags:
  - ramsey-theory
  - combinatorics
  - diffsequence
  - van-der-waerden
created: 2026-07-21
last_updated: 2026-07-21
source_count: 0
confidence: medium
status: active
---

# New Bounds on Diffsequences

**作者**：Alexander Clifton（Institute for Basic Science, Daejeon, South Korea）

**出处**：arXiv:2110.10760v4 [math.CO], 2022 年 12 月

## 核心贡献

本论文研究 [[ramsey-theory|Ramsey 理论]] 中的 [[diffsequence|diffsequence]]（差序列）问题。对于正整数集合 D，一个 $k$ 项 D-diffsequence 是满足相邻项之差落在 D 中的递增正整数序列 $a_1 < a_2 < \cdots < a_k$。论文给出了 $\Delta(D,k)$（保证任意 2-染色包含单色 k 项 D-diffsequence 的最小整数 $n$）的三个主要结果。

**定理 1.1**：当 $D = \{2^i \mid i \in \mathbb{Z}_{\ge 0}\}$ 时，证明了指数下界 $\Delta(D,k) \ge 2^{\sqrt{2k}} + \cdots$，验证了 Chokshi, Clifton, Landman, and Sawin (2018) 的猜想。此前 Landman 和 Robertson (2003) 证明了线性上界 $2k-1$。

**命题 1.2**：阶乘集合 $D = \{k! \mid k \in \mathbb{Z}^+\}$ 不是 2-accessible（即存在染色避免任意长的单色 D-diffsequence）。该证明使用基于 Beatty 序列的染色 $\lfloor n\alpha\rfloor$ 的奇偶性，构造性地展示存在长度至多为 3 的单色 diffsequence。

**定理 1.3**：完全刻画了乘积集合 $D_{\{a_n\}} = \{\prod_{i=1}^k a_i \mid k \in \mathbb{Z}^+\}$（其中 $a_i \ge 2$ for $i \ge 2$）的 2-accessibility：$D_{\{a_n\}}$ 是 2-accessible 当且仅当 $\{a_n\}$ 包含任意长的连续 2 的串。该定理统一了幂次 2（2-accessible）和阶乘（not 2-accessible）作为特例。

## 方法

证明技术包括：(1) 基于 Thue-Morse 序列构造周期性染色族 $P_t$ 和精炼染色族 $P_{t,u}$，对 gaps 大小的分布进行归纳组合分析获得指数下界；(2) 使用 Beatty 序列染色（$\lfloor n\alpha\rfloor$ 奇偶性）配合精心构造的无理数 $\alpha$ 证明不可达性；(3) 嵌套区间构造法证明乘积集合的完整分类。

## 局限性与开放问题

下界指数为 $\Theta(\sqrt{k})$，而上界为 $\Theta(k)$，存在指数差距。作者猜想下界是渐近紧的（Conjecture 5.1）。另提出多个开放问题：形如 $D_{\delta,\alpha} = \{\lfloor\delta\alpha^i\rfloor\}$ 的集合的 accessibility 特性，以及是否存在增长速率条件 disqualify 某个集合是 2-accessible 的。

