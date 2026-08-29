---
title: "Laplacian Kernel Temporal Regularization（Laplacian 核时域正则）"
type: technique
tags:
  - laplacian-regularization
  - circular-convolution
  - time-series
  - fft
  - low-rank
created: 2026-08-29
last_updated: 2026-08-29
source_count: 1
confidence: medium
status: active
---

# Laplacian Kernel Temporal Regularization（Laplacian 核时域正则）

**Laplacian kernelized temporal regularization** 是 LCR 论文（arXiv:2212.01529v3）为刻画时间序列局部趋势提出的正则机制：把图 Laplacian 的平滑思想搬到时间维，用"无向 circulant 图的 Laplacian 矩阵第一列"定义一个 Laplacian kernel，并让正则项写成 circular convolution，从而整个正则可经 FFT 在频域计算[^src-lcr]。论文自述这是首个与 circular convolution 结合、从而可用 FFT 的 Laplacian 核时域正则（第 2.2 节，作者自述口径）[^src-lcr]。

## 定义（论文 Definition 1，第 4.1 节）

给定序列 $\mathbf{x}\in\mathbb{R}^T$ 与核尺寸 $\tau\in\mathbb{Z}^+$（$\tau \le \frac{1}{2}(T-1)$），Laplacian kernel 定义为

$$\boldsymbol{\ell} \triangleq (2\tau, \underbrace{-1,\cdots,-1}_{\tau}, \underbrace{0,\cdots,0}_{T-2\tau-1}, \underbrace{-1,\cdots,-1}_{\tau})^\top \in \mathbb{R}^T$$

它是度 $2\tau$ 的无向 circulant 图之 Laplacian 矩阵 $L = D - A$ 的第一列（度矩阵对角元为 $2\tau$）[^src-lcr]。论文以 5 个节点的度 2、度 4 circulant 图为例说明该构造（Fig. 1）[^src-lcr]。

## 正则形式与频域等价

时域正则定义为（式 5）：$R_\tau(\mathbf{x}) = \frac{1}{2}\|L\mathbf{x}\|_2^2 = \frac{1}{2}\|C(\boldsymbol{\ell})\mathbf{x}\|_2^2 = \frac{1}{2}\|\boldsymbol{\ell} \star \mathbf{x}\|_2^2$，其中 $\star$ 是 circular convolution，$C(\cdot)$ 是 circulant 算子[^src-lcr]。直观上它度量每个值与其相邻 $\tau$ 个值的差异之和，起局部时间平滑作用；度 $2\tau$ 的取值取决于局部依赖强度与缺失场景（第 4.1 节）[^src-lcr]。由卷积定理（Theorem 1）与 Parseval 定理，它在频域等价于（式 8）：

$$R_\tau(\mathbf{x}) = \frac{1}{2T}\|\mathcal{F}(\boldsymbol{\ell}) \circ \mathcal{F}(\mathbf{x})\|_2^2$$

论文强调这层等价使建模无需显式构造 Laplacian 矩阵，也让正则项可并入 FFT 求解框架（第 4.1 节）[^src-lcr]。

## 特例与推广（论文 Remark 2）

若把核换成有向随机游走形式的 $\boldsymbol{\ell} = (1, 0, \cdots, 0, -1)^\top$，该正则退化为二次变差（quadratic variation, QV）正则 $\frac{1}{2}\|\boldsymbol{\ell}\star\mathbf{x}\|_2^2 = \frac{1}{2}\mathbf{x}^\top \tilde{L}\mathbf{x}$（$\tilde{L}$ 为度 2 的 circulant 矩阵）（式 6，第 4.1 节）[^src-lcr]。也就是说，QV 正则是 Laplacian 核正则的度 2 有向特例。

## 循环边界假设与翻转操作（论文 Remark 1、Fig. 7）

circulant 结构隐含"序列首尾相连"的假设。论文在 Remark 1 自认这在真实数据分析中是缺点，给出两个补救：(i) 单变量情形把序列与其翻转拼接 $\mathbf{x}_{new} = [\mathbf{x}; J_T\mathbf{x}] \in \mathbb{R}^{2T}$（$J_T$ 为反对角线为 1 的交换矩阵，式 4）；(ii) 多元（LCR-2D）情形用 2N×2T 的四块翻转矩阵作为输入，再按块平均还原（第 6.1.3 节，Fig. 7）[^src-lcr]。对日周期强的数据（Portland 速度/体积、PeMS），论文说明首尾可经 circulant 结构直接相连、无需翻转（第 5、6.2 节）[^src-lcr]。

## 在 LCR 中的位置

该正则与 circulant matrix nuclear norm 组成 LCR 目标的两个部分：前者管局部趋势，后者管全局低秩；$\gamma$ 控制两者权重。机制与实验详见 [[lcr]][^src-lcr]。

## 相关页面

- [[lcr]] — 使用该正则的插补模型
- [[directed-graph-laplacian-regularizer]] — 有向图 Laplacian 正则（谱滤波语境的同类机制）
- [[sheaf-laplacian]] — 图 Laplacian 向边语义的推广（时空建模的另一支）
- [[low-rank-prior-estimation]] — LOFT 的低秩先验：同为"低秩 + 结构先验"插补，但先验经神经参数化构造
- [[fedformer]] — 频域建模的深度路线对照（频域增强注意力）

[^src-lcr]: [[source-lcr]]
