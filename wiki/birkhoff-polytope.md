---
title: "Birkhoff Polytope"
type: concept
tags:
  - doubly-stochastic-matrices
  - optimal-transport
  - geometry
  - permutation-matrices
created: 2026-06-22
last_updated: 2026-06-22
source_count: 1
confidence: high
status: active
---

# Birkhoff Polytope

Birkhoff 多面体（Birkhoff polytope）是 $n \times n$ 双随机矩阵的集合：

$$\mathcal{B}_n = \left\{ M \in \mathbb{R}^{n \times n} \mid M \geq 0,\ M\mathbf{1}_n = \mathbf{1}_n,\ \mathbf{1}_n^\top M = \mathbf{1}_n^\top \right\}$$

它是置换矩阵集合的凸包，也是概率单纯形乘积空间中的凸多面体[^src-mhc-manifold-constrained-hyper-connections]。

## Birkhoff–von Neumann 定理

该定理指出：任意双随机矩阵都可以表示为若干置换矩阵的凸组合。因此，$H_l^{res} x_l$ 可理解为对 $n$ 个残差流进行“软置换”后再加权平均，其中权重非负且归一化[^src-mhc-manifold-constrained-hyper-connections]。

## 在 mHC 中的作用

在 [[manifold-constrained-hyper-connections|mHC]] 中，残差流之间的混合矩阵 $H_l^{res}$ 被投影到 Birkhoff 多面体上。由于双随机矩阵的谱范数不超过 1 且对矩阵乘法封闭，这一约束同时实现了：
1. 单层信号非扩张；
2. 深层复合映射保持守恒；
3. 几何上可解释的跨流特征融合[^src-mhc-manifold-constrained-hyper-connections]。

mHC 采用 [[sinkhorn-algorithm|Sinkhorn-Knopp 迭代]] 将任意实矩阵指数化后交替归一化行与列，从而逼近 Birkhoff 多面体上的投影[^src-mhc-manifold-constrained-hyper-connections]。

## 相关概念

- [[sinkhorn-algorithm|Sinkhorn-Knopp 算法]]
- [[manifold-constrained-hyper-connections|Manifold-Constrained Hyper-Connections (mHC)]]

[^src-mhc-manifold-constrained-hyper-connections]: [[source-mhc-manifold-constrained-hyper-connections]]
