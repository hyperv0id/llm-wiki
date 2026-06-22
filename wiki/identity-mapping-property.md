---
title: "Identity Mapping Property in Residual Networks"
type: concept
tags:
  - residual-networks
  - deep-learning
  - optimization-stability
  - gradient-flow
created: 2026-06-22
last_updated: 2026-06-22
source_count: 1
confidence: high
status: active
---

# Identity Mapping Property in Residual Networks

残差连接的恒等映射性质指：在残差块 $x_{l+1} = x_l + F(x_l, W_l)$ 中，浅层信号 $x_l$ 可以直接、不加修改地传递到深层[^src-mhc-manifold-constrained-hyper-connections]。递归展开到第 $L$ 层可得：

$$x_L = x_l + \sum_{i=l}^{L-1} F(x_i, W_i)$$

其中 $x_l$ 即为恒等映射项[^src-mhc-manifold-constrained-hyper-connections]。

## 为什么重要

恒等映射性质为大深度网络的训练提供了稳定性[^src-mhc-manifold-constrained-hyper-connections]：
- 前向传播中，信号始终保持一个“无修改直通”分量，避免信息在深层被完全重写；
- 反向传播中，梯度可以直接从深层流回浅层，而不必经过每个残差函数，抑制梯度消失或爆炸；
- 当残差函数 $F$ 的初始化或学习动态不理想时，模型仍可退化到恒等映射，降低优化难度[^src-mhc-manifold-constrained-hyper-connections]。

## 在 Hyper-Connections 中的破坏

[[hyper-connections|Hyper-Connections]] 将残差流扩展为 $n$ 流，并把恒等项替换为可学习的 $H_l^{res} x_l$。递归展开后，浅层到深层的传递由复合映射 $\prod_i H_i^{res}$ 决定。由于 $H_l^{res}$ 无约束，该复合映射可能显著偏离恒等映射，导致信号均值不守恒、幅值无界放大或衰减，从而在大规模训练中触发不稳定[^src-mhc-manifold-constrained-hyper-connections]。

## mHC 的恢复方式

[[manifold-constrained-hyper-connections|mHC]] 通过将 $H_l^{res}$ 约束为双随机矩阵来恢复恒等映射：双随机矩阵的行和、列和均为 1，使得 $H_l^{res} x_l$ 成为输入特征的凸组合，保持均值不变；同时双随机矩阵对乘法封闭，保证任意深度的复合映射仍守恒[^src-mhc-manifold-constrained-hyper-connections]。当 $n=1$ 时，双随机条件退化为标量 1，严格恢复原始残差连接[^src-mhc-manifold-constrained-hyper-connections]。

## 相关概念

- [[manifold-constrained-hyper-connections|mHC]]
- [[hyper-connections|Hyper-Connections]]
- [[birkhoff-polytope|Birkhoff 多面体]]
- [[residual-connections-as-diffusion|Residual Connections as Euler Steps of Reverse Diffusion]]

[^src-mhc-manifold-constrained-hyper-connections]: [[source-mhc-manifold-constrained-hyper-connections]]
