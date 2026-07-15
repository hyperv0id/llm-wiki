---
title: "Source: RiverMamba"
type: source-summary
tags:
  - state-space-model
  - mamba
  - flood-forecasting
  - spatio-temporal
  - hydrology
  - neurips-2025
created: 2026-07-21
last_updated: 2026-07-21
source_count: 0
confidence: low
status: active
---

# Source: RiverMamba — A State Space Model for Global River Discharge and Flood Forecasting

**作者**：Mohamad Hakam Shams Eddin, Yikui Zhang, Stefan Kollet, Juergen Gall（University of Bonn & Research Centre Jülich）

**发表**：NeurIPS 2025

**项目页**：<https://hakamshams.github.io/RiverMamba>

## 核心贡献

RiverMamba 是首个能在 0.05° 网格上实现全球河流流量和洪水预报的深度学习方法，预报提前期最长 7 天。它引入了三个关键创新：

1. **Mamba-based 时空编码**：利用双向 Mamba 块（选择性状态空间模型）建模全球河流网络的时空关系，通过空间填充曲线（Sweep 和 Gilbert）将采样点串行化为 1D 序列，线性复杂度下维持超大感受野，覆盖亚马逊级别的完整河网。
2. **Hindcast-Forecast 分层架构**：Hindcast 层编码历史再分析数据（ERA5-Land + GloFAS + CPC 降水），多层逐步时间下采样压缩至单向量；Forecast 层顺序集成 ECMWF HRES 气象预报，每层处理单步 lead time 并通过空间近邻信息校正气象强迫的不确定性。
3. **LOAN + 洪水加权损失**：Location-Aware Adaptive Normalization 将静态河流属性（集水区形态）注入特征归一化；损失权重按洪水重现期 r 缩放，补偿罕见极端事件的样本不均衡。

## 实验与性能

在 GloFAS 再分析（1979-2018 训练/2021-2024 测试）和 GRDC 实测站点（3366 站）上评估。RiverMamba 在所有指标（R²、KGE、F1）和所有 lead time（1-7 天）上超越 Persistence、Climatology、GloFAS 物理模型和 LSTM baseline。整体平均 F1：GloFAS 再分析 0.4589 vs LSTM 0.3582；GRDC 实测 0.2427 vs LSTM 0.1475。消融实验验证了洪水加权损失、空间填充曲线选择和气象强迫串行集成的必要性。

## 局限

ERA5-Land 实际延迟 5 天（论文假设 1 天），需替换为近实时再分析数据；GRDC 受人为干预（大坝/灌溉）影响，模型在当前框架下无法学习这些不可观测的人类水管理行为；缺少不确定性量化，预报误差来源需进一步分析。
