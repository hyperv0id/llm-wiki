---
title: "MCR² (Maximal Coding Rate Reduction)"
type: concept
tags:
  - representation-learning
  - interpretability
  - compression
  - subspace-learning
  - coding-theory
created: 2026-04-29
last_updated: 2026-04-29
source_count: 1
confidence: medium
status: active
---

# MCR² (Maximal Coding Rate Reduction)

**MCR²**（Maximal Coding Rate Reduction，最大编码率降低）是 Yu 等人提出的一种学习紧凑且结构化表示的优化目标，通过最大化编码率降低来实现子空间聚类效果[^src-cbsa]。

## 核心思想

MCR² 的目标是：给定一组数据点，学习一个表示使得**同类点被压缩到同一低维子空间，不同类点被展开到不同子空间**——即学习一个**子空间并集（union of subspaces）**的表示[^src-cbsa]。

## 数学定义

### 编码率 (Coding Rate)

编码率衡量在给定量化精度 ε 下，描述一个数据分布所需的比特数。对于中心化数据 $Z \in \mathbb{R}^{d \times N}$，编码率定义为：

$$R(Z) = \frac{d}{2} \log \det\left(I_N + \frac{1}{N\epsilon^2} Z^\top Z\right)$$

直观理解：
- 数据分布越紧凑 → 编码率越低（压缩效率高）
- 数据分布越分散 → 编码率越高（需要更多比特描述）[^src-cbsa]

### MCR² 目标

$$\max_{U^{[K]}} \Delta R(Z | U^{[K]}) = R(Z) - \sum_{k=1}^{K} R(U_k^\top Z) - \lambda\|Z\|_0$$

其中：
- $R(Z)$：**展开项**（expansion term），在整体 ambient 空间中编码，**防止坍缩**
- $R(U_k^\top Z)$：**压缩项**（compression term），将 token 投影到第 k 个子空间后编码，**鼓励紧凑表示**
- $\lambda\|Z\|_0$：稀疏性惩罚，稀疏编码
- $U^{[K]} = \{U_k\}_{k=1}^K$：K 个 p 维子空间的正交基[^src-cbsa]

## 直观解释

### 压缩 vs 展开

| 项 | 作用 | 目标 |
|----|------|------|
| $R(Z)$ (展开) | 在 ambient 空间编码 | 保持信息，避免所有点塌到原点 |
| $\sum_k R(U_k^\top Z)$ (压缩) | 在子空间编码 | 同类点压缩到低维子空间 |

**目标**：同类点 → 低编码率（紧凑）；不同类 → 高编码率差（分离）[^src-cbsa]。

### 子空间视角

- 每个子空间 $U_k$ 对应一个类别/簇
- 投影 $U_k^\top Z$ 丢弃了与子空间正交的所有信息
- 压缩项鼓励数据在子空间内紧凑排列

## 在 CBSA/CRATE 中的应用

### MSSA 的导出

Yu 等人证明：MCR² 目标对输入 token 的**压缩项**的**梯度步骤**对应于一种可解释的 softmax attention（MSSA）：

$$\text{MSSA}(Z | U^{[K]}) = \sum_k U_k^\top \text{softmax}(U_k^\top Z Z^\top U_k) U_k^\top Z$$

这个 attention 矩阵直接来自 Gram 矩阵 $U_k^\top Z Z^\top U_k$，因此复杂度为 $O(N^2)$[^src-cbsa]。

### CBSA 的改进

MSSA 的问题：Gram 矩阵导致 $O(N^2)$ 复杂度，无法处理长序列。

**CBSA 的核心改进**：引入**代表性 token** $Q = q(Z)$，用 $m \ll N$ 个代表替换压缩项中的所有 N 个 token：

$$\max_Z R(Z) - \sum_k R(U_k^\top Q) - \lambda\|Z\|_0 \quad \text{s.t. } |R(U_k^\top Q) - R(U_k^\top Z)| \leq \tau$$

这使得：
1. 压缩目标可高效计算（只需 m 个代表 vs N 个 token）
2. 通过算法展开可导出线性复杂度的 CBSA
3. 通过改变代表的结构，可实例化不同注意力机制[^src-cbsa]

## 与其他目标的对比

### 对比 InfoNCE / contrastive loss

- InfoNCE：对比学习目标，鼓励正样本相似、负样本分离
- **MCR²**：子空间聚类目标，鼓励同类压缩到子空间、不同类展开到不同子空间
- MCR² 不需要负样本，更适合无监督聚类

### 对比 PCA / autoencoder

- PCA：线性降维，最小化重构误差
- **MCR²**：非线性子空间学习，最大化类间分离 + 类内紧凑

## 理论性质

### 1. 渐近行为

当 $N \to \infty$ 且 $\epsilon \to 0$ 时，编码率与数据分布的**对数行列式**相关，渐近为：

$$R(Z) \approx \sum_{i=1}^d \log \sigma_i$$

即编码率与奇异值的对数成正比，反映数据在主成分方向上的能量分布[^src-cbsa]。

### 2. 子空间恢复

在理想条件下（足够的样本、正确的子空间维度），MCR² 目标可以**精确恢复**ground-truth 子空间结构[^src-cbsa]。

## 引用

[^src-cbsa]: [[source-cbsa]]