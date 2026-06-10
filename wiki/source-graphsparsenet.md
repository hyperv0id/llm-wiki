---
title: "GraphSparseNet: A Novel Method for Large Scale Traffic Flow Prediction"
type: source-summary
tags:
  - spatiotemporal
  - traffic-forecasting
  - gnn
  - scalability
  - large-scale
created: 2026-06-10
last_updated: 2026-06-10
source_count: 0
confidence: high
status: active
---

# GraphSparseNet — Source Summary

**Authors**: Weiyang Kong, Kaiqi Wu, Sen Zhang, Yubao Liu (Sun Yat-Sen University)
**Venue**: PVLDB 18(7): 2295–2307, 2025
**Code**: <https://github.com/PolynomeK/GSNet>

## 核心贡献

GraphSparseNet (GSNet) 针对 GNN 在大规模交通数据上的可扩展性问题，核心观察是：训练良好的自适应邻接矩阵高度稀疏——只有少量节点对有意义的连接——但现有方法仍在学习完整的 N×N 矩阵，导致 O(N²) 复杂度。

GSNet 的核心创新：
1. **理论证明**（Theorem 3.1）：秩为 C 的邻接矩阵可完全由两个小矩阵 K（C×C 低维邻接）和 U（组合系数）在低维空间中等价表达，无需学习完整 N×N 矩阵。
2. **双模块架构**：Feature Extractor（基于节点嵌入的压缩-解压管道，学习节点特征）+ Relational Compressor（压缩→拼接系数矩阵 U→低维邻接 K 特征融合→解压）。两个模块均为 O(N) 线性复杂度。
3. **压缩空间对齐**：两个模块的压缩维度 C 保持一致，确保低维空间中节点特征与邻接关系的无缝融合。

## 实验结果

- 4 个数据集（PEMS07 883节点 / PEMS08 170节点 / England 314节点 / CA 8,600节点）
- 对比 13 个基线（ARIMA 到 UniST）
- CA 数据集上 MAE=19.76（SOTA），其余数据集竞争性或最优
- CA 上训练速度比 BigST 快 3.51×，比 GWNet/AGCRN 快 64–70×
- 消融实验：移除 Relational Compressor 或 K 矩阵对精度影响最大

## 局限

- 较小数据集上精度优势不明显（C 值受限）
- C 值与精度/效率之间存在 trade-off
- 仅验证于交通领域，其他时空预测任务的泛化性有待检验
