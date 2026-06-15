---
title: "Topology-Aware Node Embedding"
type: technique
tags:
  - graph
  - node-embedding
  - laplacian
  - inductive-learning
  - spatial-temporal
created: 2026-06-15
last_updated: 2026-06-15
source_count: 1
confidence: high
status: active
---

# Topology-Aware Node Embedding

Topology-Aware Node Embedding（拓扑感知节点嵌入）是 [[std-plm|STD-PLM]] (AAAI 2025) 提出的归纳式节点表示方法，基于图拉普拉斯矩阵的特征向量，使 PLM 能够理解并利用空间-时间数据的拓扑结构，且具备**跨图结构的归纳学习能力**[^src-std-plm]。

## 设计动机

节点嵌入需要满足两个需求[^src-std-plm]：
1. 反映每个节点的静态特征（度、连通性等拓扑属性）
2. 具备跨不同图结构的归纳学习能力（即不同图间可迁移）

图拉普拉斯矩阵恰好满足这两个需求：
- 拉普拉斯矩阵包含图的度、连通性等关键结构信息
- 其特征向量相互正交，能有效区分不同节点

## 构造方法

### 特征向量选择

对归一化拉普拉斯矩阵 $L = I - D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$ 做特征分解 $L = V\Lambda V^{-1}$，选择前 $K$ 个最大特征值对应的特征向量 $V' \in \mathbb{R}^{N \times K}$[^src-std-plm]：

```
w* = argtopK(diag(Λ))
V' = V[:, w*]
```

选择 $K \ll N$（而非全部 $N$ 个特征向量）使得嵌入维度与图阶数 $N$ 解耦，实现跨图转移[^src-std-plm]。

### 线性映射

```python
E_N = W_ne @ V' + b_ne  # W_ne ∈ R^{K×dn}, b_ne ∈ R^{dn}
```

经广播得到最终嵌入 $E_N \in \mathbb{R}^{T \times N \times dn}$，其中 $W_{ne}$ 和 $b_{ne}$ 为可训练参数[^src-std-plm]。

## 与周期嵌入的配合

STD-PLM 的 Spatial-Temporal Embedding 模块由两类嵌入组成，共同支撑后续的 [[spatial-temporal-tokenizer|Spatial-Temporal Tokenizer]][^src-std-plm]：

| 嵌入 | 来源 | 作用 |
|------|------|------|
| **拓扑感知节点嵌入** | 拉普拉斯特征向量 + 线性层 | 编码节点静态属性与图结构 |
| **周期感知时间嵌入** | 288-dim（day）+ 7-dim（week）字典 | 编码时间周期性 |

## Connections

- 论文：[[std-plm|STD-PLM]] — 嵌入所属的框架
- 核心组件：[[spatial-temporal-tokenizer]] — 空间 token 中 $Z_{intrinsic}$ 的构造基础
- 对比：[[node-embedding-regularization]] — 抑制自适应图学习中节点嵌入的过参数化
- 相关：[[sheaf-laplacian]] — 层拉普拉斯算子，图拉普拉斯在胞腔层框架中的推广

[^src-std-plm]: [[source-std-plm]]
