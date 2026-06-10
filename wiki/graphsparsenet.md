---
title: "GraphSparseNet (GSNet)"
type: entity
tags:
  - spatiotemporal
  - traffic-forecasting
  - gnn
  - scalability
  - lightweight
created: 2026-06-10
last_updated: 2026-06-10
source_count: 1
confidence: high
status: active
---

# GraphSparseNet (GSNet)

GraphSparseNet（GSNet）是由中山大学提出的面向大规模交通预测的可扩展 GNN 框架（PVLDB 2025）[^src-graphsparsenet]。其核心思想是：既然训练良好的自适应邻接矩阵高度稀疏，就不必学习完整的 N×N 矩阵——可以在远小于 N 的低维压缩空间中建模节点关系。

## 动机

现有 GNN 方法的可扩展性瓶颈在于自适应邻接矩阵的 O(N²) 空间和时间复杂度[^src-graphsparsenet]：

- **分解方法**（如 GWNet 的 Tucker 分解）：仅减少参数量，不降低计算复杂度
- **稀疏化方法**（如 AGS）：仅在推理阶段有效，训练阶段仍需全矩阵
- **核方法**（如 BigST）：梯度异常值影响训练稳定性

GSNet 的观察：训练良好的邻接矩阵中，绝大多数节点的加权度很低——只有少量节点对有强连接。这意味着学习完整 N×N 邻接矩阵是极大的浪费[^src-graphsparsenet]。

## 架构

GSNet 由两个 O(N) 线性复杂度模块组成，堆叠 L 层，通过 skip connection 连接：

### Feature Extractor (FE)

- 输入：拼接的输入嵌入 P 和节点嵌入 Q
- 压缩：V₁ = W₁Q + B₁ → 投影到 C 维空间
- 激活：SoftMax
- 解压：V₂ = W₂Q + B₂ → 还原到原始维度
- 目的：学习节点特征 + 为 RC 提供压缩参考

### Relational Compressor (RC)

- 压缩：V₃ = W₃(P||Q) + B₃ → 投影到 C 维空间
- 拼接系数矩阵 U：H' = H || U
- 低维特征融合：H'' = K · H'（K 是 C×C 低维自适应邻接矩阵）
- 激活 + 解压：V₄ = W₄(P||Q) + B₄ → 还原

## 关键特性

| 特性 | 说明 |
|------|------|
| 时间复杂度 | O(N) 线性 |
| 空间复杂度 | O(N) 线性 |
| 训练速度 (CA) | BigST 的 3.51×，GWNet 的 64× |
| 预测精度 (CA) | MAE 19.76（SOTA，8,600 节点） |
| C 值 trade-off | 越大精度越高，但显存和耗时增加 |

[^src-graphsparsenet]: [[source-graphsparsenet]]
