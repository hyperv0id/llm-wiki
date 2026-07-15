---
title: "Directed Graph Laplacian Regularizer (DGLR)"
type: technique
tags:
  - graph-signal-processing
  - directed-graph
  - regularization
  - spectral-filtering
created: 2026-07-16
last_updated: 2026-07-16
source_count: 1
confidence: medium
status: active
---

# Directed Graph Laplacian Regularizer (DGLR)

**DGLR**（$\ell_2$ 范数有向图 Laplacian 正则项）是一种用于量化信号在有向图上平滑程度的变分项，由 Qi et al. (ICML 2026) 在混合图交通预测的工作中提出[^src-lightweight-mixed-graph-unrolling]。

## 定义

给定有向图 $G^d$，其行随机游走邻接矩阵 $\mathbf{W}_r^d = (\mathbf{D}^d)^{-1}\mathbf{W}^d$ 作为图平移算子（GSO），DGLR 定义为信号 $\mathbf{x}$ 与其图平移版本的 $\ell_2$ 距离[^src-lightweight-mixed-graph-unrolling]：

$$\text{DGLR}(\mathbf{x}) = \|\mathbf{x} - \mathbf{W}_r^d \mathbf{x}\|_2^2 = \|\mathbf{L}_r^d \mathbf{x}\|_2^2 = \mathbf{x}^\top \underbrace{(\mathbf{L}_r^d)^\top \mathbf{L}_r^d}_{\triangleq \mathbf{L}^{dr}} \mathbf{x}$$

其中 $\mathbf{L}_r^d = \mathbf{I} - \mathbf{W}_r^d$ 是有向随机游走 Laplacian，$\mathbf{L}^{dr} = (\mathbf{L}_r^d)^\top \mathbf{L}_r^d$ 是对称且半正定（PSD）的对称化有向图 Laplacian。

## 关键性质

1. **对称化后 PSD**：$\mathbf{L}_r^d$ 虽然不对称，但其对称化版本 $\mathbf{L}^{dr}$ 是 PSD，可以特征分解获得频率解释[^src-lightweight-mixed-graph-unrolling]。

2. **零频率信号是常数向量**：$\mathbf{1}^\top \mathbf{L}^{dr} \mathbf{1} = 0$，因为 $\mathbf{W}_r^d$ 行随机。这符合直觉：常数信号是最平滑的。

3. **低通滤波解释**：DGLR 作为目标函数时的解是对 $\mathbf{L}^{dr}$ 的低通滤波输出，频率响应 $f(\xi) = (1 + \frac{2\mu_{d,2}}{\rho_d}\xi)^{-1}$。

4. **退化到无向图**：对于无权的有向路径图（带源节点自环），DGLR 等价于无向线图的 GLR（Theorem 3.1）[^src-lightweight-mixed-graph-unrolling]。

## 与 GLR 的关系

| 特性 | GLR（无向） | DGLR（有向） |
|------|-----------|------------|
| Laplacian | $\mathbf{L}^u$ (对称) | $\mathbf{L}^{dr} = (\mathbf{L}_r^d)^\top \mathbf{L}_r^d$ (对称化) |
| 零频率 | $\mathbf{1}^\top \mathbf{L}^u \mathbf{1} = 0$ | $\mathbf{1}^\top \mathbf{L}^{dr} \mathbf{1} = 0$ |
| 距离度量 | 平方差 $(x_i - x_j)^2$ | 子节点与父母的加权差 |

DGLR 的关键创新在于：不依赖有向 Laplacian 的特征分解（非对称矩阵无谱定理保证），而是通过变分项构造实现对有向图信号平滑性的量化和促进[^src-lightweight-mixed-graph-unrolling]。

## 与 DGTV 的配合

DGLR（$\ell_2$）与 [[directed-graph-total-variation|DGTV]]（$\ell_1$）的组合形成 elastic net 正则化[^src-lightweight-mixed-graph-unrolling]：DGLR 做全局谱收缩，DGTV 做逐坐标软阈值（局部衰减），两者共同将信号推向零频率集 $\{\mathbf{x} \mid \mathbf{x} = \mathbf{W}_r^d \mathbf{x}\}$。

## 参考文献

[^src-lightweight-mixed-graph-unrolling]: [[source-lightweight-mixed-graph-unrolling]]
