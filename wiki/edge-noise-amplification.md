---
title: "Edge Noise Amplification"
type: concept
tags:
  - adaptive-graph
  - graph-learning
  - spatiotemporal-forecasting
  - activation-function
created: 2026-07-19
last_updated: 2026-07-19
source_count: 1
confidence: medium
status: active
---

# Edge Noise Amplification

**边噪声放大**（Edge Noise Amplification）是 MAGE 揭示的[[source-gwnet|GWNet]]式自适应图学习（A=Softmax(ReLU(E₁E₂ᵀ))）中的关键缺陷[^src-mage]。

## 机制

在传统自适应图学习中，邻接矩阵构造为 A = Softmax(ReLU(E₁E₂ᵀ))。ReLU 函数 f(x)=max(0,x) 对输入实施非对称变换[^src-mage]：

- **负值**：本应表示节点间负相关/不相关的条目，通过 Softmax 归一化后本会被压制。但 ReLU 将负值截断为 0，这使得原本"负相关"的节点对在 Softmax 中获得了与微弱正相关节点相当的地位，从而**人为放大了噪声边**[^src-mage]。
- **正值**：被 ReLU 保留，但相对于噪声边的对比度降低。Softmax 的归一化特性使得大量 0 值边稀释了正边权，**压制真实信号**[^src-mage]。

MAGE 的理论分析（Appendix A.1）给出了严格的数学刻画。解决方案是直接移除 ReLU，使用 A = Softmax(E₁E₂ᵀ)，再引入 kernel 近似解决由此带来的计算挑战[^src-mage]。

## 影响

边噪声放大导致自适应图学习到的是被污染的空间拓扑——本不该存在的伪依赖被模型当作真实空间关系加以利用，损害预测准确性和泛化能力[^src-mage]。

[^src-mage]: [[source-mage]]
