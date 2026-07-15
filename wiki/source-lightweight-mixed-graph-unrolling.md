---
title: "Lightweight and Interpretable Transformer via Mixed Graph Algorithm Unrolling for Traffic Forecast"
type: source-summary
tags:
  - traffic-forecasting
  - algorithm-unrolling
  - graph-signal-processing
  - lightweight-model
  - interpretability
  - transformer
  - spatial-temporal
created: 2026-07-16
last_updated: 2026-07-16
source_count: 0
confidence: medium
status: active
---

# Lightweight and Interpretable Transformer via Mixed Graph Algorithm Unrolling for Traffic Forecast

**Authors:** Ji Qi, Mingxiao Liu, Tam Thuc Do, Yuzhe Li, Zhuoshi Pan, Gene Cheung, H. Vicky Zhao (Tsinghua University & York University)
**Venue:** ICML 2026
**Code:** <https://github.com/SingularityUndefined/Unrolling-GSP-STForecast>

## 核心贡献

本文提出了一种通过混合图算法展开构建的轻量级、可解释的类 Transformer 神经网络，用于交通预测。核心思想是将一个混合图（无向 + 有向）优化算法通过 ADMM 展开为前馈网络层。

## 方法要点

1. **混合图建模**：构建两种图 — 无向图 $G^u$ 捕获空间相关性，有向无环图 $G^d$ 捕获时间顺序关系。$G^d$ 的边从时间 $\tau$ 的节点指向未来窗口内 $\tau+1,\dots,\tau+W$ 的同一节点。

2. **有向图正则项**：设计两个新的变分项量化信号在有向图上的平滑性 — DGLR（$\ell_2$ 范数，$\|\mathbf{x} - \mathbf{W}_r^d\mathbf{x}\|_2^2$）和 DGTV（$\ell_1$ 范数，$\|\mathbf{x} - \mathbf{W}_r^d\mathbf{x}\|_1$），两者结合形成类似 elastic net 的正则化。

3. **ADMM 展开**：将优化目标（保真项 + GLR + DGLR + DGTV）通过 ADMM 框架交替求解，每一步（CG 求解、软阈值、乘子更新）展开为神经网络的一层，共 5 个 ADMM block、每 block 25 层。

4. **图学习即自注意力**：定期插入的图学习模块（UGL/DGL）通过马氏距离计算节点间边权重，数学形式上等价于自注意力中的 scaled dot-product + softmax，但参数远少于 Q/K/V 矩阵。

5. **低通滤波解释**：展开的每一层操作可解释为对混合图 Laplacian 低通滤波器的参数化实现。

## 实验结果

在 PEMS03 和 METR-LA 上，模型参数量仅 38K（论文声称 PDFormer 的 7.2%；实际 38/1404≈2.7%），推理计算量仅 PDFormer 的 4.9%（0.087/1.771 GFLOPs），同时在 60 分钟预测中至少一个指标进入 Top-3。消融实验验证了 DGLR 和 DGTV 各自的重要性，以及有向时间建模优于无向。
