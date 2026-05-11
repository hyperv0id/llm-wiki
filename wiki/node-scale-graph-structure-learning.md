---
title: "Node-Scale Graph Structure Learning"
type: technique
tags:
  - graph-structure-learning
  - spatio-temporal
  - data-imputation
  - feature-heterogeneity
created: 2026-05-11
last_updated: 2026-05-11
source_count: 1
confidence: medium
status: active
---

# Node-Scale Graph Structure Learning

Node-Scale Graph Structure Learning 是 [[gsli|GSLI]] 的核心模块之一，通过为每个特征独立学习全局元图来捕获特征特定的空间依赖，解决时空数据中的特征异质性问题[^src-yang-gsli-2025]。

## 核心思想

现有方法使用统一图结构建模所有特征的空间关系，但不同特征（如风向 DD 和风速 FH）在同一对站点间的空间相关性不同。Node-scale 学习为每个特征 $f$ 构建独立的元图 $\dot{G}_f^\Omega$，避免异质特征间的信息干扰[^src-yang-gsli-2025]。

## 方法

### 元节点嵌入与显著度建模

对每个特征 $f$ 分配两对可学习元节点嵌入：
- 源节点嵌入 $\Omega_1^f \in \mathbb{R}^{N \times d}$
- 目标节点嵌入 $\Omega_2^f \in \mathbb{R}^{N \times d}$

通过显著度建模增强高影响力节点的边权重：

$$P_\Omega^f = \text{MLP}(\Omega_1^f)$$
$$\tilde{\Omega}_1^f = \Omega_1^f \odot P_\Omega^f$$

### 元图构建

$$\dot{A}_\Omega^f = \text{SoftMax}(\text{ReLU}(\tilde{\Omega}_1^f {\Omega_2^f}^\top))$$

- ReLU 消除弱相关边
- SoftMax 归一化邻接矩阵

### 图扩散卷积

融合给定图 $G$ 和元图 $\dot{G}_f^\Omega$ 的空间依赖：

$$R_f^{NL} = \sum_{k=0}^{K} \left[ (\dot{A}_\Omega^f)^k A R_f \Theta_{k,f}^{\Omega_1} + D_O^{-1} (A^\top)^k R_f \Theta_{k,f}^{\Omega_3} \right]$$

其中 $K$ 为扩散步数，$D_O$、$D_I$ 为出度/入度矩阵，$\Theta$ 为图卷积核[^src-yang-gsli-2025]。

## 理论保证

**Proposition 1**：标准图扩散卷积的结果 $a_{i1}^{\Omega} r_{1,f_2,c} + \cdots + x r_{j,f_2,c} + \cdots + a_{iN}^{\Omega} r_{N,f_2,c}$ 无法处理特征异质性——当 $f_1$ 和 $f_2$ 在节点 $i,j$ 间的相关性权重不同（$x \neq y$）时，统一图结构强制使用相同的权重。

**Proposition 2**：Node-scale 学习可获得期望的特征独立空间依赖——结果为 $a_{i1}^{\Omega} r_{1,f_2,c} + \cdots + y r_{j,f_2,c} + \cdots + a_{iN,f_2,c}^{\Omega} r_{N,f_2,c}$，其中权重与特征 $f_2$ 的相关性一致[^src-yang-gsli-2025]。

## 复杂度

- 时间：$O(F N^2 T C + F N T C^2 + F N^2 d + F N d^2)$
- 空间：$O(F N T C + F C^2 + F N^2 + F N d + F d^2)$

## 消融证据

移除 node-scale 学习（仅保留 feature-scale）后，CN 数据集 RMSE 从 0.253 降至 0.263（+4.0%），验证了特征独立空间建模的重要性[^src-yang-gsli-2025]。

## 关联页面

- [[gsli]] — GSLI 框架
- [[feature-scale-graph-structure-learning]] — 特征尺度图结构学习
- [[prominence-modeling-gsl]] — 显著度建模
- [[imputeformer]] — ImputeFormer（无图结构学习，使用节点嵌入）
- [[adaptive-graph-agent-attention]] — AGA-Att（大规模图的代理注意力）

[^src-yang-gsli-2025]: [[source-yang-gsli-2025]]
