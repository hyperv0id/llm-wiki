---
title: "Directed Graph Total Variation (DGTV)"
type: technique
tags:
  - graph-signal-processing
  - directed-graph
  - regularization
  - sparse-modeling
created: 2026-07-16
last_updated: 2026-07-16
source_count: 1
confidence: medium
status: active
---

# Directed Graph Total Variation (DGTV)

**DGTV**（$\ell_1$ 范数有向图总变差）是一种用于量化信号在有向图上局部变化的稀疏正则项，由 Qi et al. (ICML 2026) 在混合图交通预测的工作中提出[^src-lightweight-mixed-graph-unrolling]。

## 定义

给定以 $\mathbf{W}_r^d$ 为图平移算子（GSO）的有向图 $G^d$，DGTV 定义为[^src-lightweight-mixed-graph-unrolling]：

$$\text{DGTV}(\mathbf{x}) = \|\mathbf{x} - \mathbf{W}_r^d \mathbf{x}\|_1 = \|\mathbf{L}_r^d \mathbf{x}\|_1 = \sum_{j \in \bar{\mathcal{S}}} \left| x_j - \sum_{i} w_{j,i} x_i \right|$$

其中 $\bar{\mathcal{S}}$ 是非源节点集合。DGTV 计算每个子节点与其所有父母加权平均值之差的绝对值之和。

## 关键性质

1. **非对称**：与 [[directed-graph-laplacian-regularizer|DGLR]] 不同，DGTV 没有对称形式，因此不能简单通过特征分解获得频率解释[^src-lightweight-mixed-graph-unrolling]。

2. **两通道滤波器组解释**：在 ADMM 求解中，DGTV 通过软阈值操作实现：
   $$\phi_i = \text{sign}(\delta) \cdot \max(|\delta| - \rho^{-1}\mu_{d,1}, 0)$$
   其中 $\delta = (\mathbf{L}_r^d)_i \mathbf{x} - \rho^{-1}\gamma_i$。这等价于将 $\mathbf{L}_r^d$ 视为高通通道、$\mathbf{W}_r^d = \mathbf{I} - \mathbf{L}_r^d$ 视为低通通道的两通道滤波器组[^src-lightweight-mixed-graph-unrolling]：软阈值衰减高通通道分量，而低通通道保持不变。

3. **稀疏性促进**：$\ell_1$ 范数倾向产生稀疏梯度，即信号变化集中在少数边上——适合交通流中的局部突发事件建模[^src-lightweight-mixed-graph-unrolling]。

## 与 DGLR 的比较

| 特性 | DGLR ($\ell_2$) | DGTV ($\ell_1$) |
|------|----------------|-----------------|
| 作用方式 | 全局谱收缩 | 逐坐标局部衰减 |
| 稀疏性 | 不促进 | 促进 |
| 频率解释 | 有（对称化特征分解） | 无（仅滤波器组解释） |
| 数值稳定性 | 更好 | 需要小心调参 |

两者的组合构成 **elastic net 正则化**，在统计学习中已知具有更好的鲁棒性[^src-lightweight-mixed-graph-unrolling]。实验消融表明，移除 DGTV 或 DGLR 均导致性能明显下降。

## 参考文献

[^src-lightweight-mixed-graph-unrolling]: [[source-lightweight-mixed-graph-unrolling]]
