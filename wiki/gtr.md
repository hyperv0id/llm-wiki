---
title: "Global Temporal Retriever (GTR)"
type: entity
tags:
  - time-series-forecasting
  - periodicity
  - global-retrieval
  - plug-and-play
  - multivariate
  - iclr-2026
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: medium
status: active
---

# Global Temporal Retriever (GTR)

GTR（Global Temporal Retriever）是一个轻量级、即插即用的多变量时间序列预测增强模块，由 HKUST-GZ / HKUST / 山东大学团队提出，发表于 ICLR 2026。[^src-gtr]

## 动机

现有 MTSF 模型受限于固定回溯窗口，当真实周期长度远大于观测历史时，全局周期模式对模型不可见。[^src-gtr] 在 Electricity 数据集上的 Pearson 相关性分析揭示了一个关键发现：跨越全局周期的片段相关性（如 Corr(S₁₂, S₅)=0.96）往往高于邻近局部片段（如 Corr(S₁₂, S₁₃)=0.94）——全局依赖比局部近邻更具预测价值。[^src-gtr] 直接扩展回溯窗口会导致过拟合、计算成本膨胀以及冗余信息难以区分。

## 核心机制

GTR 维护一个可学习的全局参数矩阵 Q ∈ R^(L×N)，其中 L 为全局周期长度，N 为变量数。[^src-gtr] 该矩阵在训练过程中自动学习各变量在整个周期内的模式结构。

GTR 处理流程分为三步：[^src-gtr]

1. **周期信息对齐（Cycle Information Alignment）**：对于起始绝对时间为 t₀ 的输入序列，计算周期索引向量 i = (t₀ mod L) + τ mod L（τ=0,1,...,T-1），据此从 Q 中检索对应段并通过线性映射得到全局时间参考 qₙ。

2. **时间模式提取（Temporal Pattern Extraction）**：将输入 xₙ 与全局参考 qₙ 堆叠为 2×T 矩阵 Fₙ，通过 2D 卷积（核宽度由高频局周期长度 P 决定，如日模式在小时数据中 P=24）提取跨局部-全局尺度的时间模式 hₙ。

3. **残差融合**：zₙ = xₙ + Dropout(hₙ)，将增强表示送入主干模型。

GTR 的关键优势在于：无论回溯窗口多短，模型都能通过绝对时间位置检索整个周期的信息。[^src-gtr]

## 主干模型

采用轻量 MLP 架构：线性输入投影后将 Z 映射到隐空间 R^(D×N)（D=512），通过两层 GeLU 激活的线性层加残差连接，再经 Dropout 后线性输出投影到预测长度。[^src-gtr] 使用 RevIN（Reversible Instance Normalization）稳定非平稳性。[^src-gtr]

## 效率

GTR 模块仅 40.1K 参数、4.50M MACs。完整系统（GTR + MLP）0.98M 参数，为 [[itransformer|iTransformer]] 的 19.0%，单 epoch 训练时间 22.3 秒（Electricity 数据集，T=96，S=720）。[^src-gtr] 总复杂度 O(NT² + Nd² + NTd + NSd)，对 N 和 S 线性。

## 实验结果

在 6 个真实数据集（ETT 系列、Electricity、Traffic、Solar-Energy、Weather、PEMS 系列）上评估，与 RAFT、S-Mamba、[[tqn|TQNet]]、TimeXer、[[cyclenet|CycleNet]]、SOFTS、[[timemixer|TimeMixer]]、[[itransformer|iTransformer]]、[[patchtst|PatchTST]]、[[ltsf-linear|DLinear]] 等基线对比。[^src-gtr]

**长期预测**：T=96 固定回溯，16 个预测任务中 10 个 top-2。Solar-Energy 上 MSE 比 CycleNet 降低 8.2%。[^src-gtr]

**短期预测**（PEMS 系列）：8 个任务全部 top-2。PEMS03 上 MSE 比 iTransformer 降低 28.7%，比 S-Mamba 降低 15.7%。[^src-gtr]

**跨模型泛化**：GTR 作为插件可显著提升多种架构：[^src-gtr]
- [[itransformer|iTransformer]]：PEMS03 MSE 降低 62.2%，PEMS04 降低 37.9%
- [[patchtst|PatchTST]]：PEMS04 MSE 降低 56.2%
- [[ltsf-linear|DLinear]]：PEMS04 MSE 降低 91.9%

**回溯窗口鲁棒性**：GTR 在所有窗口长度下均优于基线，最短窗口时优势最大。基线模型在窗口减小时 MSE 指数增长，GTR 保持稳定。[^src-gtr]

## 理论分析

论文从贝叶斯估计角度证明：在观测噪声方差 σ²_η 大于全局嵌入误差方差 σ²_ε 的条件下，GTR 融合后的变量间相关性估计误差严格小于原始观测的估计误差。[^src-gtr] 这解释了为何 GTR 能拉近模型学到的多变量相关性与全局相关结构之间的距离（图 4 可视化验证）。

## 局限性

1. **固定周期长度假设**：假定单一全局周期长度，不适用于时变周期的数据。[^src-gtr]
2. **跨通道共享周期**：所有变量使用相同周期长度，不适用于异质周期的多变量场景。[^src-gtr]
3. **长周期计算负担**：2D 卷积核宽度随 P 线性增长，O(NTP) 复杂度。输入序列长时线性投影 O(NT²) 成为瓶颈。[^src-gtr]

## 与其他方法的关系

- 不同于 [[autoformer|Autoformer]]、[[fedformer|FEDformer]] 的季节-趋势分解和频域建模，GTR 通过显式的全局周期嵌入直接检索而非隐式推断周期结构。[^src-gtr]
- 与 [[cyclenet|CycleNet]] 的可学习循环周期相似，但 GTR 按绝对位置检索而非学习残差周期；GTR 在 Solar-Energy 上 MSE 比 CycleNet 降低 8.2%。[^src-gtr]
- 与 [[timesnet|TimesNet]] 的 1D→2D 变换共享将时间序列视为 2D 结构的思路，但 GTR 的 2D 用于融合局部与全局参考，而非捕捉周期内和跨周期变化。[^src-gtr]
- 与 [[phat|PHAT]] 的周期性建模侧重异质周期的分桶策略，GTR 则侧重全局检索。[^src-gtr]
- **[[rast|RAST]]**（AAAI 2026）同为检索增强的预测方法，但 RAST 在时空双维度（时间+空间）执行 FAISS 向量检索，且面向交通预测场景；GTR 仅在时间维度通过可学习参数 $Q$ 进行全局周期检索。[^src-gtr]
- **[[retrieval-guidance|Retrieval Guidance]]**（MiDDiR, ICLR 2026 under review）类似地使用检索增强生成，但采用不同的机制：MiDDiR 在推理时检索训练样本并分析性偏置扩散得分函数，而非将检索结果作为模型输入特征。GTR 在训练时就学习检索模块，MiDDiR 则训练后仅推理时检索。[^src-gtr]

## 关键参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| L | 全局周期长度 | 数据集相关 |
| P | 高频局周期长度 | 日模式 P=24（小时级数据） |
| D | 隐层维度 | 512 |
| T | 回溯窗口 | 96（默认） |
| S | 预测长度 | {96, 192, 336, 720} |

[^src-gtr]: [[source-gtr]]
