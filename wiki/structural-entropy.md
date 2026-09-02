---
title: "结构熵与编码树 (Structural Entropy & Encoding Tree)"
type: concept
tags:
  - graph
  - information-theory
  - community-detection
  - hierarchical-clustering
created: 2026-09-02
last_updated: 2026-09-02
source_count: 1
confidence: medium
status: active
---

# 结构熵与编码树（Structural Entropy & Encoding Tree）

结构信息理论（论文转述 Li & Pan, IEEE Transactions on Information Theory 2016）把 Shannon 熵推广到图数据，度量图中蕴含的不确定性与信息，并据此获得有信息量的层次结构做图压缩。理论由两部分组成：编码树与结构熵[^src-multispans]。

## 编码树（Encoding Tree）

对图 G = {V, E}，编码树 T 是把图压缩编码的层次结构：每个树节点 α 关联一个顶点子集 T_α ⊆ V；非叶节点的孩子把它的顶点集划分成互不相交的子集（孩子按从左到右编号 α⟨i⟩）。编码树由此把图抽象并编码为层次化的社区结构[^src-multispans]。

## 结构熵（Structural Entropy）

结构熵由编码树与图共同决定：$H^T(G) = \sum_{\alpha \in T, \alpha \neq \lambda} H^T(G;\alpha)$，其中 $H^T(G;\alpha) = -\frac{g_\alpha}{vol(G)} \log_2 \frac{V_\alpha}{V_{\alpha^-}}$；g_α 是从树节点 α 的顶点集外部连入内部的边权之和，vol(G) 是全图总度数，V_α 是 T_α 内的顶点度数和，V_α₋ 是父节点的度数和。最小化图结构熵的编码树压缩了最多的图知识——把图的总信息量视为常数时，它是最优的本质层次结构表示[^src-multispans]。

## 最小化算法

MultiSPANS 沿用 deDoc 的启发式算法与树算子（combination 算子与 merge 算子）：从只有一层的平铺编码树（所有叶节点都是根的直接孩子）出发，每轮贪心选择并执行使结构熵下降最多的节点对与算子，直到熵不再持续下降，得到最优编码树 T*[^src-multispans]。

## 作为图的层次聚类

编码树本质上是一种拓扑驱动的层次聚类：树有 L 层，就有 L 个不同粒度的图节点划分。与 Louvain/METIS 的单层划分不同（见 [[graph-node-clustering]]），编码树一次给出全部粒度，且层数在优化中自适应确定（论文称取决于图的规模与结构复杂度）。MultiSPANS 把每层划分直接变成一层注意力掩码；其消融显示结构熵导出的掩码优于用最小熵式层次社区检测 Infomap 构造的掩码（完整机制去掉后 PEMSD4-flow MAE 19.07→19.73，换成 Infomap 掩码则 19.43）[^src-multispans]。

## 相对结构熵

论文进一步定义相对结构熵以刻画树节点与其子结构间的相对复杂度和信息量：两个叶节点间的相对结构熵经由它们的最低公共祖先 θ 分段求和（式 10），等价于把编码树视为图后、两叶间最短有向路径上相连节点的相对结构熵之和。MultiSPANS 用它生成层级相关矩阵作为空间注意力的相对位置编码[^src-multispans]。

## 在机器学习中的应用（论文 related work 转述）

结构信息理论最早用于网络安全（网络抗毁/安全性指标）与生物信息学（Hi-C 数据的 TAD 解码、肿瘤亚型界定、皮肤病变分割）；近期工作把结构熵用于改进 GNN——超参选择、图结构学习（如同组作者的 SE-GSL, WWW 2023）、层次池化——以及强化学习的角色抽象与状态抽象[^src-multispans]。MultiSPANS 自述是首次把结构熵理论用于优化（时空 Transformer 的）空间注意力机制[^src-multispans]。

## 相关页面

[[multispans]] · [[graph-node-clustering]] · [[traffic-forecasting]]

[^src-multispans]: [[source-multispans]]
