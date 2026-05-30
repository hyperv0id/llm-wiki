---
title: "GRIN"
type: entity
tags:
  - graph-neural-network
  - spatio-temporal
  - data-imputation
  - message-passing
  - iclr-2022
created: 2026-05-30
last_updated: 2026-05-30
source_count: 1
confidence: medium
status: active
---

# GRIN

**GRIN** (Graph Recurrent Imputation Network) 是由 Cini, Marisca & Alippi (USI IDSIA / Politecnico di Milano) 于 ICLR 2022 提出的首个基于图神经网络的多元时间序列填补模型。通过消息传递机制显式建模传感器间的空间依赖关系，在 4 个基准数据集上 MAE 改善常超 20%[^src-2108-00298]。

## 核心架构

```
输入 X[t,t+T] + Mask M[t,t+T]
        │
  ┌─────┴──────┐
  │ Forward     │ Backward
  │ Module      │ Module
  │  ├─ MPGRU   │  ├─ MPGRU      ← GRU 门控 = 消息传递层
  │  │  Encoder │  │  Encoder
  │  └─ Spatial │  └─ Spatial    ← 两阶段空间解码
  │     Decoder │     Decoder
  └─────┬──────┘
        │ 拼接表示
        ▼
     MLP → 最终填补
```

### MPGRU（Message-Passing GRU）

将标准 GRU 的门控操作替换为消息传递层，在更新隐藏状态的同时聚合邻居信息[^src-2108-00298]：

- 重置门 $r_t^i = \sigma(\text{MPNN}(\hat{x}_t^{(2)} \| m_t^i \| h_{t-1}^i, W_t))$
- 更新门 $u_t^i = \sigma(\text{MPNN}(\hat{x}_t^{(2)} \| m_t^i \| h_{t-1}^i, W_t))$
- 候选状态 $c_t^i = \tanh(\text{MPNN}(\hat{x}_t^{(2)} \| m_t^i \| r_t^i \odot h_{t-1}^i, W_t))$

使用扩散卷积（diffusion convolution）作为消息传递算子。

### 空间解码器的关键设计

**仅邻居约束**：空间解码器的 MPNN 在计算节点 $i$ 的填补表示时，仅聚合来自邻居 $j \in \mathcal{N}(i) / \{i\}$ 的消息，**排除节点自身**。这迫使模型必须从空间依赖（而非自身特征）推断缺失值，产生正则化效果[^src-2108-00298]。

### Filler 算子

$$\Phi(Y_t) = M_t \odot X_t + \bar{M}_t \odot Y_t$$

在已知位置保留观测值，仅在缺失位置替换为预测值。所有阶段共用此算子确保观测数据不被覆盖。

## 关键性能

| 数据集 | vs BRITS | 参数量 |
|--------|----------|--------|
| AQI (out-of-sample) | MAE ↓20%+ | ~200K |
| METR-LA Block | MAE ↓29% | vs BRITS ~4M |
| PEMS-BAY Point | MAE ↓50% | |
| CER-E Block | MAE ↓36% | |
| 合成粒子 | MSE ↓11-14× | |

GRIN 参数量仅 ~200K，远小于 BRITS 的 ~4M，在参数效率上显著更优。

## 消融实验

| 变体 | AQI MAE | METR-LA Block MAE |
|------|---------|-------------------|
| GRIN (完整) | **14.73** | **2.03** |
| w/o 空间解码器 | 15.40 (+4.5%) | 2.32 (+14.3%) |
| w/ 去噪解码器 | 17.23 (+17.0%) | 2.96 (+45.8%) |
| MPGRU (单向) | 18.76 (+27.3%) | 2.57 (+26.6%) |

空间解码器对块缺失场景贡献最大；去噪式解码器在块缺失下显著退化（+45.8%）。

## 虚拟感知能力

GRIN 的空间解码器归纳偏置使其可用于虚拟感知（virtual sensing / kriging）：遮蔽训练集传感器，模型从零重建其时序。AQI-36 实验显示对最高/最低连接度传感器分别取得 MAE 11.74 / 20.00[^src-2108-00298]。

## 与后续工作的关系

| 方向 | 方法 | 相对 GRIN 的改进 |
|------|------|-----------------|
| 图结构学习 | [[gsli]] (AAAI 2025) | 为每个特征独立学习元图，解决固定图限制 |
| 低秩 Transformer | [[imputeformer]] (KDD 2024) | 低秩归纳偏置替代 GNN，训练快 15× |
| 扩散填补 | [[cofill]] (2025) | 非递归扩散解决误差累积 |

GRIN 作为 GNN 填补的开山之作，确立了空间解码和消息传递填补的基本范式，但其固定图结构和自回归误差累积的局限催生了后续改进。

## 局限性

1. 自回归结构在长时间缺失下误差累积
2. 依赖预定义图结构（阈值高斯核 / kNN）
3. 仅在平稳过程假设下评估
4. 扩散卷积的 O(E) 复杂度在大规模图上有开销

## 关联页面

- [[message-passing-imputation]] — 消息传递填补范式
- [[spatial-imputation-decoder]] — 空间解码器的仅邻居约束设计
- [[mpgru]] — Message-Passing GRU 单元
- [[imputeformer]] — ImputeFormer，低秩 Transformer 填补
- [[cofill]] — CoFILL，条件扩散填补
- [[gsli]] — GSLI，多尺度图结构学习填补
- [[traffic-forecasting]] — 交通预测

[^src-2108-00298]: [[source-2108-00298]]
