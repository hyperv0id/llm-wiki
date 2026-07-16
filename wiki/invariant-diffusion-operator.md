---
title: "Invariant Diffusion Operator"
type: technique
tags:
  - pde
  - attention
  - green-function
  - spatio-temporal-forecasting
created: 2026-07-23
last_updated: 2026-07-23
source_count: 1
confidence: medium
status: active
---

# Invariant Diffusion Operator

Invariant Diffusion Operator 是 [[stpde|STPDE]] 框架中负责捕获**跨环境共享的普适物理传输机制**的核心组件[^src-stpde]。它以拉普拉斯算子 $\nabla^2$ 为原型，通过 Green 函数与注意力机制的理论对应来实现。

## 理论基础

### PDE 形式化

STPDE 将潜在场 $u(x, \tau)$ 建模为非齐次扩散过程：

$$\frac{\partial u(x, \tau)}{\partial \tau} = \alpha(\Phi, \xi) \nabla^2 u(x, \tau) + S(\Phi, \xi)$$

其中 $\nabla^2$ 是**域不变**的拉普拉斯算子（捕获拓扑感知的传输与守恒），$\alpha$ 和 $S$ 由 [[environment-basis-manifold|Environment Basis Manifold]] 条件化[^src-stpde]。

### Green 函数 → 全局注意力（Theorem 4.1）

扩散方程的解可通过 Green 函数（热核）表示为全局积分变换：

$$u(x, \tau) = \int_\Omega K(x, x'; \tau) \, u(x', 0) \, dx'$$

在离散图上，热核矩阵 $\mathbf{K}(\tau) = \exp(-\tau \mathbf{L})$ 的传播可通过 **row-wise 归一化**后近似为 softmax 注意力权重[^src-stpde]：

$$\alpha_{ij} = \frac{\exp(\mathbf{h}_{q,i}^\top \mathbf{h}_{k,j} / \sqrt{D})}{\sum_z \exp(\mathbf{h}_{q,i}^\top \mathbf{h}_{k,z} / \sqrt{D})}$$

### 线性复杂度实现（Theorem 4.2）

通过可分离核假设 $\mathbf{K}_{ij} \approx \phi_q(\mathbf{h}_{q,i})^\top \phi_k(\mathbf{h}_{k,j})$，利用结合律将计算重组为**聚合–分发**两步：

$$\mathbf{G} = \sum_{j=1}^N \phi_k(\mathbf{h}_{k,j}) \mathbf{h}_{v,j}^\top \in \mathbb{R}^{D \times D}$$

$$\mathbf{h}'_i = \mathbf{G}^\top \phi_q(\mathbf{h}_{q,i})$$

复杂度从 $O(N^2)$ 降至 $O(ND^2)$（$D$ 固定时对 $N$ 线性）[^src-stpde]。

## 与标准注意力的关系

与标准 Transformer 自注意力不同，Invariant Diffusion Operator：
- 不依赖位置编码——扩散核天然编码了空间拓扑距离
- 使用线性注意力而非 softmax 注意力（效率优先）
- 可叠加可选的局部几何先验 $\mathbf{A}$（邻接矩阵），在 ID 场景下加速收敛

## 与 GCN 的对比

消融实验中 w/ GCN（用局部图卷积替换扩散算子）表现不及全局扩散，因为 GCN 主要依赖局部消息传递，在可比网络深度下无法实现等效的全局长程耦合[^src-stpde]。

[^src-stpde]: [[source-stpde]]
