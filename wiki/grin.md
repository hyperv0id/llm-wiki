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
last_updated: 2026-08-29
source_count: 7
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
| Primary-Auxiliary | [[past]] (PVLDB 2025) | 外部信息辅助模式增强纤维/块缺失填补 |
| 残差扩散精炼 | [[rdpi]] (AAAI 2025) | 以 GRIN 为两阶段框架的确定性初始模型并联合训练；作者报告 AQI-36 全节点遮蔽（节点 14/31）上 MAE 相对 GRIN 降低 31%/26%（Table 6）[^src-rdpi] |
| 一致性插补 | [[costi]] (KBS 2025) | 沿用 GRIN 基准（Cini et al.）的数据集、划分与种子，将 GRIN 纳入 Table 6 对比；该表五组设置中 CoSTI 的 MAE/MSE 均低于 GRIN（如 AQI-36 10.13/377.48 vs 12.08/523.14）[^src-costi] |

GRIN 作为 GNN 填补的开山之作，确立了空间解码和消息传递填补的基本范式，但其固定图结构和自回归误差累积的局限催生了后续改进。

## 综述归类

Wang & Du 等人的 MTSI 综述将 GRIN 归为预测式-GNN 类插补方法（Table 1 缺失机制标注 MCAR/MAR），称其为首个基于图的循环 MTSI 架构——用双向图循环神经网络捕获时间动态与空间相似性——并提到 SPIN 通过稀疏时空注意力机制缓解 GRIN 的误差传播、增强对数据稀疏性的鲁棒性[^src-mts-imputation-survey]。该定位与本页"首个基于 GNN 的多元时间序列填补模型"的原文口径一致。

## 局限性

1. 自回归结构在长时间缺失下误差累积
2. 依赖预定义图结构（阈值高斯核 / kNN）
3. 仅在平稳过程假设下评估
4. 扩散卷积的 O(E) 复杂度在大规模图上有开销

## 相关工作

- [[maginet|MagiNet]] (arXiv 2024) 直接批判 GRIN 的预填充 + 消息传递路线：它认为零预填充注入噪声、且 GRIN 在动态/连续缺失位置产生过平滑插值，转而用可学习缺失嵌入完全取消预填充[^src-maginet]
- [[fgti|FGTI]] (NeurIPS 2024) 将 GRIN 列入 15 个插补基线，且默认以单位阵作为 GRIN 的邻接矩阵（即不提供图先验）；其 Table 1 报告 KDD 10% 下 FGTI RMSE 0.406 vs GRIN 0.565（作者报告口径）[^src-fgti]

## 关联页面

- [[message-passing-imputation]] — 消息传递填补范式
- [[spatial-imputation-decoder]] — 空间解码器的仅邻居约束设计
- [[mpgru]] — Message-Passing GRU 单元
- [[imputeformer]] — ImputeFormer，低秩 Transformer 填补
- [[cofill]] — CoFILL，条件扩散填补
- [[gsli]] — GSLI，多尺度图结构学习填补
- [[past]] — PAST，primary-auxiliary 时空填补 (PVLDB 2025)
- [[giflow]] — GiFlow (ICML 2026)，图信息先验流匹配填补，以 GRIN 为时空 GNN 基线；作者报告 Air-36 point 20% 下 MAE 9.54 vs GRIN 9.94、RMSE 18.10 vs 19.09[^src-giflow]
- [[traffic-forecasting]] — 交通预测
- [[mts-imputation-taxonomy]] — MTSI 综述的分类框架，GRIN 归为预测式-GNN 类
- [[costi]] — CoSTI (KBS 2025)，一致性训练插补，沿用 GRIN 基准并对比 GRIN[^src-costi]
- [[fgti]] — FGTI (NeurIPS 2024)，频域条件扩散插补，以 GRIN 为基线（单位阵邻接）[^src-fgti]

[^src-2108-00298]: [[source-2108-00298]]
[^src-mts-imputation-survey]: [[source-mts-imputation-survey]]
[^src-maginet]: [[source-maginet]]
[^src-giflow]: [[source-giflow]]
[^src-rdpi]: [[source-rdpi]]
[^src-costi]: [[source-costi]]
[^src-fgti]: [[source-fgti]]
