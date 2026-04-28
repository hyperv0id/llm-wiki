---
title: "Long-Distance Correlation Preservation (LDCP)"
type: technique
tags:
  - position-encoding
  - extrapolation
  - theory
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

**远距离相关性保持 (Long-Distance Correlation Preservation, LDCP)** 是无限上下文外推的关键数学条件之一，确保远距离 token 仍能对注意力分数产生有意义的贡献[^src-vetcha-2026-towards-infinite-length-extrapolation]。

## 定义

位置编码方法满足 LDCP，当且仅当存在常数 $C > 0$ 和无限集合 $S \subset \mathbb{Z}$，使得对所有 $n \in S$：

$$|E[A(n)]| \geq C$$

或更一般地，随着 $|n|$ 增加，注意力分数的绝对值以高概率不趋近于零[^src-vetcha-2026-towards-infinite-length-extrapolation]。

## 意义

- 确保模型能够捕获长程依赖关系
- 防止注意力机制退化为只关注局部上下文

## 关键矛盾：LDCP 与收敛归一化

**引理 3.6**：若位置编码方法满足 LDCP，则它无法同时满足收敛归一化[^src-vetcha-2026-towards-infinite-length-extrapolation]。

证明：对于每个 $n \in S$，由 Jensen 不等式 $E[e^{A(n)}] \geq e^{E[A(n)]} \geq e^C$。由于 $S$ 是无限集合，$\sum_{n \in S} E[e^{A(n)}] \geq \sum_{n \in S} e^C = \infty$，导致归一化常数发散。

## 方法对比

| 方法 | LDCP | 备注 |
|------|------|------|
| RoPE | ✅ | 注意力分数保持 O(1) 量级，但导致归一化发散 |
| ALiBi | ❌ | 线性衰减 $-m\|n\|$ 完全抑制远距离相关性 |
| APE | 局部 | 仅在局部范围 $N_{LDCP}$ 内保持，远距离通过概率保持 |

## APE 的局部 LDCP

APE 通过次线性衰减（log + √|n|）实现更大的局部 LDCP 范围[^src-vetcha-2026-towards-infinite-length-extrapolation]：

$$N_{LDCP}^{APE} \geq N_{LDCP}^{ALiBi} = \lfloor C_1 / m \rfloor$$

对于 $|n| > N_{LDCP}$，APE 通过方差优势（Chebyshev 不等式）仍能以非零概率保持远距离注意力。

---

[^src-vetcha-2026-towards-infinite-length-extrapolation]: [[source-vetcha-2026-towards-infinite-length-extrapolation]]