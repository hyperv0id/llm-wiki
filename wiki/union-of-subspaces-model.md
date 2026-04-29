---
title: "Union of Subspaces Model"
type: concept
tags:
  - subspace-learning
  - representation-learning
  - clustering
  - compression
  - mcr2
created: 2026-04-29
last_updated: 2026-04-29
source_count: 1
confidence: high
status: active
---

# Union of Subspaces Model

**Union of Subspaces Model**（子空间并集模型）是 MCR² 和 CBSA 背后的核心数据假设，假设数据分布在 K 个低维线性子空间的并集上[^src-cbsa]。

## 核心假设

现实世界的数据（如自然图像）通常可以由低维结构近似：

$$\bigcup_{k=1}^{K} \mathcal{S}_k$$

其中每个 $\mathcal{S}_k$ 是一个 p 维线性子空间，K 是子空间数量，p 是每个子空间的维度[^src-cbsa]。

## 数学定义

### 子空间

设 $\mathcal{S} \subset \mathbb{R}^d$ 是一个 p 维线性子空间，由正交基张成：

$$\mathcal{S} = \{U\alpha : \alpha \in \mathbb{R}^p\}$$

其中 $U \in \mathbb{R}^{d \times p}$ 满足 $U^\top U = I_p$（列正交）[^src-cbsa]。

### Union of Subspaces

K 个不相交的 p 维子空间并集：

$$\bigcup_{k=1}^{K} \mathcal{S}_k = \{U_k \alpha : \alpha \in \mathbb{R}^p, k \in [K]\}$$

满足 $U_i^\top U_j = I_p, \forall i \neq j$（子空间正交，即不相交）[^src-cbsa]。

## 为什么需要这个假设

### 1. 可处理性

- 非线性流形（Manifold）虽然能更好地逼近任意数据，但优化困难
- 线性子空间假设足够表达许多实际模式，同时便于数学分析
- MCR² 目标可以在这个假设下精确推导[^src-cbsa]

### 2. 聚类解释

- 每个子空间对应一个类别/簇
- 子空间并集模型 ≈ 线性子空间聚类
- 同类数据点应该落在同一子空间内[^src-cbsa]

### 3. 压缩目标

- 如果数据确实来自子空间并集，压缩 token 到子空间是"正确"的操作
- 编码率 R(U_kᵀZ) 天然度量 token 在子空间内的紧凑程度

## 在 CBSA 中的应用

### 子空间基参数化

CBSA 中的每个注意力头被建模为一个 p 维子空间：

$$U^{[K]} = \{U_k \in O(d, p)\}_{k=1}^K$$

其中 O(d, p) 是 d×p 正交矩阵集合（列正交）[^src-cbsa]。

### 子空间 vs 注意力头

| 概念 | CBSA | 标准 Transformer |
|------|------|------------------|
| 注意力头数 | H = K（子空间数） | H |
| 每个头的参数 | 子空间基 U_k ∈ ℝ^(d×p) | W_Q, W_K, W_V ∈ ℝ^(d×d) |
| 可解释性 | 子空间基有明确含义（主方向） | 无（黑盒） |
| 正交性 | U_k 列正交 | W_* 无约束 |

### 子空间表示

- 输入 token Z ∈ ℝ^(d×N)
- 投影到子空间：U_kᵀZ ∈ ℝ^(p×N)
- 压缩在 p 维子空间进行（p << d）

## 假设的局限性

### 1. 不适用于所有模态

论文承认：union of linear subspaces 假设可能不适用于所有模态和任务。因此，实验仅在视觉任务上进行，排除了 NLP 任务[^src-cbsa]。

### 2. 需要子空间正交

假设子空间之间正交（$U_i^\top U_j = I_p, \forall i \neq j$），这在实践中可能过于严格。

### 3. 维度选择

子空间维度 p 需要预先指定或学习：p 太大 → 表示能力过强；p 太小 → 欠拟合。

## 与其他表示模型的对比

| 模型 | 数据假设 | 优点 | 缺点 |
|------|----------|------|------|
| **Union of Subspaces** | K 个线性子空间并集 | 可解释、可分析 | 过于简化 |
| 非线性流形 | 低维弯曲曲面 | 表达力强 | 优化困难 |
| 稀疏编码 | 字典原子线性组合 | 可解释稀疏性 | 需要过完备字典 |
| 深度表示 | 层次非线性变换 | 表达力强 | 黑盒 |

## 引用

[^src-cbsa]: [[source-cbsa]]