---
title: "LrSGAT"
type: technique
tags:
  - graph-attention
  - low-rank
  - dynamic-graph
  - spatio-temporal
  - data-imputation
  - sampling
  - arxiv-2025
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# LrSGAT (Low-rank guided Sampling Graph ATtention)

**LrSGAT** 是 [[stamimputer|STAMImputer]] (arXiv 2025) 提出的空间注意力专家机制，灵感来自 Fang et al. (2023) 的谱图注意力与 [[imputeformer|ImputeFormer]] 的低秩诱导[^src-stamimputer]。其目标是**动态平衡路网上的局部与全局空间相关性**，同时在复杂度可控的前提下，把采样得到的注意力向量复用为半自适应动态图[^src-stamimputer]。

LrSGAT 工作流分三步（对应论文 Figure 1.2 的 a/b/c）[^src-stamimputer]。

## (a) 采样投影器 (Sampling Projector)

先用一个由静态拓扑（K 近邻图）引导的通用 GAT，在局部图上算出局部注意力反馈 $E_t^G \in \mathbb{R}^{N\times E}$（$E$ 为每节点邻居数）[^src-stamimputer]。再用可训练打分向量 $W^{sc}$ 把局部注意力压成**显著度评分** $E_t^W \in \mathbb{R}^{N\times 1}$，并据此做**混合采样**得到节点集 $N_t^S = \langle N_t^T \,\|\, N_t^U\rangle$[^src-stamimputer]：

- $N_t^T$：显著度最高的 $S=\lceil \log N\rceil$ 个节点（潜在交通枢纽）[^src-stamimputer]；
- $N_t^U$：从其余节点按概率分布采样的 $S$ 个节点（保留增量随机影响）[^src-stamimputer]。

这种 "top-S + 概率采样" 的混合策略确保潜在枢纽不会因数据缺失而被遗漏[^src-stamimputer]。采样后用自注意力把 query/key 映射到样本，得到投影向量 $P_t^S$ 与投影消息 $M_t$，规模因 $S\ll N$ 而大幅缩小[^src-stamimputer]。

## (b) 低秩引导再注意力 (Re-attention, ReAT)

低秩分解将矩阵 $X\in\mathbb{R}^{m\times n}$ 近似为 $X=UV^\top$（秩 $k\ll\min(m,n)$）以压缩复杂结构[^src-stamimputer]。LrSGAT 在此层用上一层传来的投影向量 $P_t^S$（作 key，引导全局节点关注关键节点）与投影消息 $M_t$（作 value，恢复被压缩信息）做再注意力 $\hat X_t^s = \text{Atten}^P(X_t, P_t^S, M_t^S)$[^src-stamimputer]。重复注意力对高阶低秩空间矩阵反复压缩-还原，在此过程中完成缺失特征填补[^src-stamimputer]。

## (c) 半自适应动态图 (Semi-adaptive Dynamic Graph)

经典自适应邻接矩阵 $\tilde A^{adp}=\text{softmax}(\text{ReLU}(E_1 E_2^\top))$（如 Graph-WaveNet）完全由可学习参数抽象而成，没有实际节点映射，且在缺失条件下易学到低质量结构[^src-stamimputer]。LrSGAT 改用**节点支撑的采样矩阵**构建动态图——Dynamic Graph Structure Learning (DGSL)[^src-stamimputer]：

$$\tilde A_t^{adp} = \text{softmax}(\text{ReLU}(A_t^S\, E^{adp})), \quad E^{adp} = \text{toph}(M_t (E^{ref})^\top)$$

其中 $A_t^S$ 是采样注意力向量，$E^{ref}$ 是可学习的**折射向量 (refraction vector)**，$\text{toph}(\cdot)$ 把小于中位数的项置零[^src-stamimputer]。

直觉上（论文附录 A）：采样注意力近似低秩分解中的**内聚因子 (cohesion factor)**——若干关键节点的全局影响分布可视为局部社区代表；但每次采样的社区代表是无序且变化的，故引入折射向量 $E^{ref}$ 把投影消息折射为对齐的**外化因子 (extroversion factor)**，从而动态重构并精炼邻接矩阵[^src-stamimputer]。生成的动态图可即插即用地服务下游交通预测（实验中接入 Graph-WaveNet）[^src-stamimputer]。

## 复杂度与效果

LrSGAT 作为空间专家的复杂度为 $O(N\log N\, D)$，DGSL 层为 $O(N^2\log N)$[^src-stamimputer]。消融实验表明，在**块缺失**模式下将 LrSGAT 换成 MLP 会导致性能急剧退化，说明它是 STAMImputer 处理块缺失最关键的组件[^src-stamimputer]。

## 关联

- 低秩诱导思想与 [[imputeformer|ImputeFormer]] 的 [[projected-attention]]/[[embedded-attention]] 一脉相承，但 LrSGAT 用显式节点采样而非纯可学习 embedding agents[^src-stamimputer]。
- 动态图构建可对比 [[gsli|GSLI]] 的多尺度图结构学习与 [[mtgnn]]/[[gwnet]] 的自适应邻接矩阵。

[^src-stamimputer]: [[source-stamimputer]]
