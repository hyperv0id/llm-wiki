---
title: "Key Normalization"
type: technique
tags:
  - transformer
  - attention
  - normalization
  - stability
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# Key Normalization

键归一化（Key Normalization）是一种注意力机制改进技术，通过对键向量进行 ℓ2 归一化来消除键范数对注意力的不成比例影响[^src-quest]。

## 背景：键范数的问题

在标准注意力机制中：
$$A_i = \text{softmax}\left(C\|q_i\|\|\bar{k}_j\|(\bar{q}_i \cdot \bar{k}_j)\right)$$

键范数 $\|k_j\|$ 直接影响注意力分数，导致：
- 高范数键"窃取"全局注意力
- 即使余弦相似度不高，高范数键也能获得高注意力
- 训练时键范数不受控制地增长[^src-quest]

## 键归一化的效果

通过 $\ell2$ 归一化键向量 $\bar{k}_j = k_j / \|k_j\|$：
$$A_i = \text{softmax}\left(C\|q_i\|(\bar{q}_i \cdot \bar{k}_j)\right)$$

效果：
- 注意力排名完全由余弦相似度决定
- 消除键范数的"窃取"效应
- 每个查询独立控制锐度（通过查询范数）[^src-quest]

## 与查询归一化的对比

| 方法 | 效果 | 局限性 |
|------|------|--------|
| Q 归一化 | 稳定训练 | 限制表达能力 |
| K 归一化 | 稳定训练 + 保持锐度控制 | 每个查询可独立调整 |
| Q、K 均归一化 | 非常稳定 | 所有查询锐度相同（QKNorm）[^src-quest] |

## 数学性质

### 注意力分布
- 归一化后，注意力分数仅取决于向量对齐
- 消除了幅度差异带来的偏差
- 保持余弦相似度的排序性质[^src-quest]

### 梯度影响
- 键的梯度不再受键范数影响
- 打破 Q-K 之间的交叉依赖
- 减少注意力熵崩溃的风险[^src-quest]

## 应用

- **QUEST 注意力**：核心机制是仅归一化键[^src-quest]
- **QKNorm**：同时归一化查询和键
- **Elliptical Attention**：结合椭球度量和键归一化[^src-quest]

## 引用

[^src-quest]: [[source-quest]]