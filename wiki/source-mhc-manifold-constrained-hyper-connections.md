---
title: "mHC: Manifold-Constrained Hyper-Connections"
type: source-summary
tags:
  - residual-connections
  - hyper-connections
  - manifold-constraint
  - large-language-models
  - training-stability
  - deepseek
created: 2026-06-22
last_updated: 2026-06-22
source_count: 1
confidence: high
status: active
---

# mHC: Manifold-Constrained Hyper-Connections

> Xie, Z., Wei, Y., Cao, H., et al. (2026). mHC: Manifold-Constrained Hyper-Connections. *arXiv:2512.24880v2 [cs.CL]*. DeepSeek-AI.

## 核心论点

[[hyper-connections|Hyper-Connections]] 通过加宽残差流提升模型拓扑复杂度，但无约束的学习映射破坏了残差连接的恒等映射性质，导致大规模训练不稳定和显存访问开销[^src-mhc-manifold-constrained-hyper-connections]。mHC 将残差映射 $H_l^{res}$ 投影到 [[birkhoff-polytope|Birkhoff 多面体]]（双随机矩阵流形），恢复信号守恒并兼顾跨流信息交换[^src-mhc-manifold-constrained-hyper-connections]。

## 方法

单层更新为 $x_{l+1} = H_l^{res} x_l + H_l^{post\top} F(H_l^{pre} x_l, W_l)$，其中 $H_l^{res}$ 经 [[sinkhorn-algorithm|Sinkhorn-Knopp]] 约束为双随机矩阵，$H_l^{pre}, H_l^{post}$ 经 Sigmoid 保证非负[^src-mhc-manifold-constrained-hyper-connections]。双随机性带来谱范数不超过 1、乘法封闭、复合映射守恒三个性质[^src-mhc-manifold-constrained-hyper-connections]。

## 效率与实验

mHC 使用 TileLang 核融合、选择性重计算与扩展 DualPipe 调度，$n=4$ 时大规模训练仅增加约 6.7% 时间开销[^src-mhc-manifold-constrained-hyper-connections]。在 3B/9B/27B 的 DeepSeek-V3 风格 MoE 模型上，27B mHC 最终损失比基线低 0.021，下游 benchmark 普遍优于基线与 HC；复合映射最大增益从 HC 的近 3000 降至约 1.6[^src-mhc-manifold-constrained-hyper-connections]。

## 展望

mHC 为探索其他几何约束的宏观架构设计提供了框架，有望推动下一代基础模型拓扑演化[^src-mhc-manifold-constrained-hyper-connections]。

[^src-mhc-manifold-constrained-hyper-connections]: [[source-mhc-manifold-constrained-hyper-connections]]
