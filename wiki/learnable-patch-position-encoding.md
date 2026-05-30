---
title: "Learnable Patch Position Encoding"
type: technique
tags:
  - time-series
  - positional-encoding
  - patch-embedding
  - cross-variate
created: 2026-05-30
last_updated: 2026-05-30
source_count: 1
confidence: medium
status: active
---

# Learnable Patch Position Encoding

CVPE 中使用的可学习位置编码 $W_P \in \mathbb{R}^{P \times d_m}$，在 patch embedding 后添加到每个 patch token，编码 patch 在时间和变量维度上的相对位置信息 [^src-cvpe-2025]。

## 机制

给定 patch embedding $\hat{X}_P \in \mathbb{R}^{N \times P \times d_m}$（N 为变量数，P 为 patch 数），添加可学习位置编码 [^src-cvpe-2025]：

$$X_P = \hat{X}_P + W_P$$

其中 $W_P \in \mathbb{R}^{P \times d_m}$ 对所有变量的所有 patch 共享，使同一时间位置的 patch 在不同变量间具有相同的偏置。

## 设计考量

- **共享 vs 变量特异**：$W_P$ 仅沿时间维度参数化，对所有变量共享。这使模型能通过位置编码标识"同一时间步的不同变量 patch"，为后续 Router-Attention 提供跨变量聚合的锚点 [^src-cvpe-2025]
- **可学习 vs 固定**：$W_P$ 是可学习的（而非如 sinusoidal 编码那样固定），允许模型自适应数据中 patch 位置的相对重要性

## 与传统位置编码的对比

| 编码类型 | 参数化 | 维度覆盖 | 可学习 |
|----------|--------|---------|--------|
| Sinusoidal (Transformer 原版) | 固定 | 仅时间 | 否 |
| ALiBi [[alibi]] | 固定斜率 | 仅距离 | 否 |
| RoPE [[yarn]] | 固定频率 | 仅位置 | 部分 |
| CVPE $W_P$ | 可学习 | 时间+跨变量上下文 | 是 |

CVPE 的 $W_P$ 虽仅沿时间维度参数化，但在聚合为 $\mathbb{R}^{N \times P \times d_m}$ 后，它为 Router-Attention 提供了"哪些 patch 处于同一时间步"的信号，间接编码了跨变量上下文 [^src-cvpe-2025]。

## 相关技术

- 关系：[[cvpe]] — 位置编码是 CVPE 的前置组件
- 关系：[[router-attention-for-cvpe]] — 位置编码后的 embedding 是 Router-Attention 的输入
- 对比：[[alibi]] — ALiBi 用加性偏置编码距离，CVPE 用可学习编码编码 patch 位置
- 对比：[[patch-based-tokenization]] — 位置编码附加在 patch tokenization 之后

[^src-cvpe-2025]: [[source-cvpe-2025]]