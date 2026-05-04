---
title: "数学直觉系列（一）：缩放因子1/√dₖ"
type: source-summary
tags:
  - transformer
  - attention
  - scaling-factor
  - numerical-stability
  - softmax
created: 2026-05-04
last_updated: 2026-05-04
source_count: 1
confidence: medium
status: active
---

# 数学直觉系列（一）：缩放因子 $1/\sqrt{d_k}$

**来源**: bluuuuue | 小红书 (2025)
**链接**: https://www.xiaohongshu.com/discovery/item/69f2e246000000001f001fd3
**类型**: 技术教程文章，2364 字

## 核心论点

本文将 Scaled Dot-Product Attention 中的缩放因子 $1/\sqrt{d_k}$ 重新定位为注意力机制的**数值稳定性条件**，而非调参技巧。文章围绕三个问题展开：不施加缩放时注意力机制在什么条件下失效？为什么恰好是 $\sqrt{d_k}$？缩放与 Softmax 饱和特性之间的数学关系是什么？[^src-bluuuuue-scaling-factor-intuition]

## 核心贡献

1. **方差膨胀的严格推导**：设 $q_i, k_i \sim N(0,1)$ 且独立，点积 $Z = \sum_{i=1}^{d_k} q_i k_i$ 的方差 $Var(Z) = d_k$，标准差 $\sigma_Z = \sqrt{d_k}$。$d_k$ 越大，点积值分布越分散。[^src-bluuuuue-scaling-factor-intuition]

2. **Softmax 饱和机制分析**：当输入方差极大时，Softmax 输出趋近 One-hot 分布。梯度公式 $\frac{\partial s_i}{\partial z_j} = s_i(\delta_{ij} - s_j)$ 在饱和态下趋近于零，导致梯度消失。[^src-bluuuuue-scaling-factor-intuition]

3. **$1/\sqrt{d_k}$ 的三个数学特性**：(a) 选择 $1/\sqrt{d_k}$ 而非 $1/d_k$——后者导致过缩放，方差变为 $1/d_k$ 趋于零；(b) 缩放保持 argmax 不变，仅改变分布形态，属于纯数值稳定性修正；(c) 对深层结构中注意力稳定性有持续贡献，而非仅在初始化阶段发挥作用。[^src-bluuuuue-scaling-factor-intuition]

## 局限性

- 假设 $Q$ 和 $K$ 各分量独立且服从 $N(0,1)$，这在训练中后期并不严格成立
- 未讨论除 $\sqrt{d_k}$ 外的其他缩放策略（如可学习温度参数）的效果对比
- 未提供实证消融实验数据，论证主要基于数学推导

## 引用

[^src-bluuuuue-scaling-factor-intuition]: [[source-bluuuuue-scaling-factor-intuition]]
