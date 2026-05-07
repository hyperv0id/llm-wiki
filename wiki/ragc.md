---
title: "RAGC"
type: entity
tags:
  - traffic-forecasting
  - spatial-temporal
  - graph-neural-network
  - scalability
  - regularization
created: 2026-05-07
last_updated: 2026-05-07
source_count: 1
confidence: high
status: active
---

# RAGC

RAGC (Regularized Adaptive Graph Convolution) 是一种用于大规模路网交通预测的模型，由 Wu, Kong, Zhang, Chen 和 Liu（中山大学）于 2026 年提出[^src-ragc-efficient-traffic-forecasting]。它在四个 LargeST 数据集上始终取得最优预测精度，同时保持有竞争力的计算效率。

## 核心问题

RAGC 针对自适应图学习的两个关键局限[^src-ragc-efficient-traffic-forecasting]：

1. **O(N²) 计算复杂度** — 自适应邻接矩阵的图卷积需要构建 N×N 矩阵，大规模路网（8,600 节点）不可扩展
2. **节点嵌入过参数化** — 节点嵌入占模型参数的主要比例（如 GWNet 占 72%，AGCRN 占 86%），但缺乏正则化

## 模型架构

RAGC 由三个组件构成[^src-ragc-efficient-traffic-forecasting]：

### 1. 嵌入层
- 输入序列 → 全连接层映射为高维表示
- 时间嵌入：time-of-day（$D_{tid}$）和 day-of-week（$D_{diw}$）字典
- 节点嵌入 $E_{node} \in \mathbb{R}^{N \times d_{node}}$，经 [[stochastic-shared-embedding|SSE]] 正则化后得到 $\tilde{E}_{node}$
- 拼接：$H^{(0)} = E_{in} \| E_{tid} \| E_{diw} \| \tilde{E}_{node}$

### 2. 自适应图卷积编码器
每层包含三个操作：
- **MLP + 残差连接**：特征交互和非线性变换，$H^{(l)}_{mlp} = \text{FC2}(\text{ReLU}(\text{FC1}(H^{(l-1)}))) + H^{(l-1)}$
- **[[efficient-cosine-operator|ECO]] 图卷积**：$H^{(l)}_g = \sum_{z=0}^{Z} A^{(z)}_{adp} H^{(l)}_{mlp} W^{(z)}_g$，线性复杂度
- **[[residual-difference-mechanism|残差差分]]**：$H^{(l)} = H^{(l)}_{mlp} - H^{(l)}_g$，抑制 SSE 噪声

### 3. 回归层
两个分支求和：$\hat{X} = \text{FC}_{node}(H^{(L)}) + \text{FC}_{global}(H_{skip})$

## 关键技术创新

| 技术 | 功能 | 复杂度 |
|------|------|--------|
| [[efficient-cosine-operator\|ECO]] | 余弦相似度图卷积 | O(N) |
| [[stochastic-shared-embedding\|SSE]] | 节点嵌入正则化 | — |
| [[residual-difference-mechanism\|RDM]] | 抑制 SSE 噪声传播 | — |

## 实验结果

在 LargeST 四个数据集上与 12 个基线模型比较[^src-ragc-efficient-traffic-forecasting]：

| 数据集 | 节点数 | RAGC MAE | 次优模型 | 改进 |
|--------|--------|----------|----------|------|
| SD | 716 | 16.16 | PatchSTG 16.90 | 4.4% |
| GBA | 2,352 | 18.33 | PatchSTG 19.50 | 6.0% |
| GLA | 3,834 | 17.75 | PatchSTG 18.96 | 6.4% |
| CA | 8,600 | 16.40 | PatchSTG 17.35 | 5.5% |

效率排名：训练速度第 2 快，推理速度第 3 快（仅次于 GSNet 和 BigST，但 GSNet 精度较差）。

## 消融实验

移除任何核心组件均导致显著精度下降[^src-ragc-efficient-traffic-forecasting]：
- **w/o SSE**：节点嵌入过拟合，验证损失明显上升
- **w/o RDM**：SSE 噪声影响训练稳定性，验证损失波动大（最低 15.55 vs RAGC 更低）
- **w/o AGC**：欠拟合，缺少空间聚合
- **w/ Dropout**：破坏空间信息完整性
- **w/ Laplacian**：依赖预定义图，缺乏数据驱动灵活性

## 相关页面

- [[efficient-cosine-operator|ECO]] — 线性复杂度图卷积算子
- [[stochastic-shared-embedding|SSE]] — 随机共享嵌入正则化
- [[residual-difference-mechanism|RDM]] — 残差差分噪声抑制
- [[node-embedding-regularization]] — 节点嵌入正则化概念
- [[traffic-forecasting]] — 交通预测方法概览
- [[large-scale-spatial-temporal-graph]] — 大规模时空图预测挑战

[^src-ragc-efficient-traffic-forecasting]: [[source-ragc-efficient-traffic-forecasting]]
