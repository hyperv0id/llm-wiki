---
title: "Hyper-Connections (HC)"
type: technique
tags:
  - residual-connections
  - macro-design
  - large-language-models
  - deep-learning
created: 2026-06-22
last_updated: 2026-06-22
source_count: 1
confidence: high
status: active
---

# Hyper-Connections (HC)

Hyper-Connections 是一种宏观网络设计范式，通过加宽残差流的宽度并在多个并行流之间引入可学习连接，在不改变单层计算量（FLOPs）的情况下提升残差网络的拓扑复杂度与表达能力[^src-mhc-manifold-constrained-hyper-connections]。

## 单层形式

HC 将第 $l$ 层的输入/输出从 $C$ 维扩展为 $n \times C$ 维的隐藏矩阵 $x_l \in \mathbb{R}^{n \times C}$，并定义：

$$x_{l+1} = H_l^{res} x_l + H_l^{post\top} F(H_l^{pre} x_l, W_l)$$

其中：
- $H_l^{pre} \in \mathbb{R}^{1 \times n}$ 从 $n$ 流残差中聚合出 $C$ 维输入；
- $H_l^{post} \in \mathbb{R}^{1 \times n}$ 将层输出映射回 $n$ 流残差；
- $H_l^{res} \in \mathbb{R}^{n \times n}$ 直接在残差流之间进行信息混合[^src-mhc-manifold-constrained-hyper-connections]。

由于 $n$ 通常远小于 $C$，这三个映射引入的额外计算可忽略，因此 HC 可在不增加 FLOPs 的前提下把残差流的信息容量与层输入维度解耦[^src-mhc-manifold-constrained-hyper-connections]。

## 动态映射与静态映射

HC 中的系数由输入相关的动态部分和全局可学习的静态部分组成。具体地，对 RMSNorm 后的 $\tilde{x}_l$ 做线性投影，再通过 tanh 与可学习门控因子生成：

$$H_l^{pre} = \alpha_l^{pre} \cdot \tanh(\theta_l^{pre} \tilde{x}_l^\top) + b_l^{pre}$$

$post$ 与 $res$ 映射同理。门控因子初始化很小，使模型在训练初期近似标准残差连接[^src-mhc-manifold-constrained-hyper-connections]。

## 组件贡献

消融实验显示，在 HC 的三个映射中，$H_l^{res}$ 对性能提升贡献最大：禁用 $H_l^{res}$（用单位矩阵替代）带来的损失下降最为明显，说明残差流内部的有效信息交换是 HC 收益的关键来源[^src-mhc-manifold-constrained-hyper-connections]。

## 规模化风险

当网络加深或规模增大时，无约束的 $H_l^{res}$ 会导致严重不稳定。多层复合映射 $\prod_{i=1}^{L-l} H_{L-i}^{res}$ 不再保持恒等映射：实验观察到最大行和/列和增益峰值可达约 3000，引发损失尖峰和梯度爆炸，限制了 HC 在大模型训练中的可用性[^src-mhc-manifold-constrained-hyper-connections]。

## 与 mHC 的关系

[[manifold-constrained-hyper-connections|Manifold-Constrained Hyper-Connections (mHC)]] 在保留 HC 拓扑扩展能力的同时，将 $H_l^{res}$ 投影到双随机矩阵流形上，从而恢复恒等映射并解决规模化训练中的不稳定问题[^src-mhc-manifold-constrained-hyper-connections]。

[^src-mhc-manifold-constrained-hyper-connections]: [[source-mhc-manifold-constrained-hyper-connections]]
