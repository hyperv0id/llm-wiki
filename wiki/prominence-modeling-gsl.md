---
title: "Prominence Modeling (GSL)"
type: technique
tags:
  - graph-structure-learning
  - spatio-temporal
  - data-imputation
  - attention-weighting
created: 2026-05-11
last_updated: 2026-05-11
source_count: 1
confidence: medium
status: active
---

# Prominence Modeling (GSL)

Prominence Modeling 是 [[gsli|GSLI]] 中的权重增强机制，在图结构学习过程中为不同节点和特征分配不同的重要性权重，使对填补贡献更大的节点/特征在元图中获得更强的边权重[^src-yang-gsli-2025]。

## 核心思想

在时空填补中，不同节点和特征对填补缺失值的贡献不同。例如，数据质量高、缺失率低的站点比缺失率高的站点更可靠；与其他特征相关性强的特征对填补贡献更大。Prominence Modeling 通过 Hadamard 积调整源嵌入的权重，使高影响力的源节点/特征产生更强的连接[^src-yang-gsli-2025]。

## 方法

### Node-Scale Prominence

对每个特征 f 的源节点嵌入计算显著度向量：

P_Omega_f = MLP(Omega_1_f)

tilde{Omega}_1_f = Omega_1_f * P_Omega_f

其中 * 为 Hadamard 积。这样，高影响力节点的边在元图中获得更强的权重。

### Feature-Scale Prominence

对所有特征的源特征嵌入计算显著度向量：

P_Phi = MLP(Phi_1)

tilde{Phi}_1 = Phi_1 * P_Phi

异质性较低、对其他特征填补贡献更大的特征获得更强的源嵌入权重[^src-yang-gsli-2025]。

## 设计动机

论文通过交叉特征自注意力机制提取的平均注意力分数发现，不同站点对整体填补的影响力不同。这启发了为节点和特征分配不同显著度的设计。

## 消融证据

移除显著度建模后（w/o Prominence），各数据集性能均有下降：

| 数据集 | GSLI RMSE | w/o Prom RMSE | 变化 |
|--------|-----------|---------------|------|
| DutchWind | 0.4101 | 0.4132 | +0.8% |
| BeijingMEO | 0.3986 | 0.4041 | +1.4% |
| LondonAQ | 0.2720 | 0.2809 | +3.3% |
| CN | 0.2534 | 0.2595 | +2.4% |

显著度建模在 LondonAQ 和 CN 数据集上贡献较大，这两个数据集的缺失率较高（13.81% 和 25.3%），验证了在高缺失率场景下区分节点/特征影响力的重要性[^src-yang-gsli-2025]。

## 与其他注意力权重机制的关系

Prominence Modeling 可视为一种轻量级的注意力权重机制，与以下方法相关但有本质区别：

- **[[adaptive-graph-agent-attention|AGA-Att]]**：FaST 中的代理注意力，通过可学习代理 token 降低空间复杂度
- **[[embedded-attention|ImputeFormer 嵌入注意力]]**：通过节点嵌入线性投影计算空间相关性
- **[[key-normalization|Key Normalization]]**：QUEST 中的键归一化，消除键范数对注意力的"窃取"效应

Prominence Modeling 不同之处在于：它直接作用于元图的源嵌入，通过 Hadamard 积显式增强高影响力节点/特征的边权重，而非通过注意力分数的归一化或代理机制[^src-yang-gsli-2025]。

## 关联页面

- [[gsli]] — GSLI 框架
- [[node-scale-graph-structure-learning]] — 节点尺度图结构学习
- [[feature-scale-graph-structure-learning]] — 特征尺度图结构学习
- [[adaptive-graph-agent-attention]] — AGA-Att 代理注意力
- [[embedded-attention]] — ImputeFormer 嵌入注意力

[^src-yang-gsli-2025]: [[source-yang-gsli-2025]]
