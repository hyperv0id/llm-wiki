---
title: "RiverMamba"
type: entity
tags:
  - state-space-model
  - mamba
  - flood-forecasting
  - spatio-temporal
  - hydrology
  - neurips-2025
created: 2026-07-21
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

# RiverMamba

**RiverMamba** 是首个能在全球 0.05° 网格上实现河流流量和洪水预报的深度学习模型，由 University of Bonn 和 Research Centre Jülich 联合开发，发表于 NeurIPS 2025[^src-rivermamba]。它基于选择性状态空间模型 [[mamba|Mamba]] 构建，实现线性复杂度的全球尺度时空建模，预报提前期达 7 天。

## 架构设计

RiverMamba 由三部分组成[^src-rivermamba]：

### 输入与嵌入

模型输入包括：ERA5-Land 再分析、GloFAS 再分析流量、CPC 统一降水分析（以上作为初始条件），以及 ECMWF HRES 气象预报。静态河流属性（LISFLOOD 集水区形态）通过 [[location-aware-adaptive-normalization|LOAN]] 层注入。预测目标是相对于前一天日均流量的变化 ΔX。

### Hindcast 层

3 层 Hindcast 块编码历史序列（T=4 天逐步下采样至 T=1）。每层包含三个核心组件：

- **序列化/反序列化**：通过 [[space-filling-curves|空间填充曲线]]（Sweep→Sweep transposed→Gilbert→Gilbert transposed 交替）将 2D 空间点转为 1D 序列，时间维度通过连接相邻时间步的曲线端点实现。四条曲线交替使用确保从不同空间视角扫描。
- **LOAN**：使用静态河流属性生成位置相关的自适应仿射参数，条件化特征归一化。
- **双向 Mamba 块**：沿序列化路径做 forward + backward 选择性 SSM 扫描，经 SiLU 门控融合。

### Forecast 层

L 个 Forecast 块（L=7，对应 1-7 天 lead time），结构与 Hindcast 相同但额外串联 HRES 气象预报。第 l 块处理 X^(t+l)_HRES，确保气象强迫与初始条件的时序关系不被打乱。回归头以 MLP 实现，融合当前块特征与其他块特征的拼接。

## 训练策略

在约 P 个全球采样点上训练，先用 GloFAS 再分析训练、再在 GRDC 实测站点微调。损失为加权 MSE，权重由两部分构成[^src-rivermamba]：

- **洪水重现期权重**：ŵ = r（若 r > 1，r 为统计重现期 1.5-500 年），否则 ŵ = 1
- **Lead time 衰减权重**：û = e^(α(L−l+1))，α = 0.25，短 lead time 权重更高

## 关键性能

在 GloFAS 再分析上：R² 0.8728、KGE 0.9125、F1 0.4589（整体平均），在所有 lead time（1-7 天）上超越 Persistence、Climatology、GloFAS 物理模型和 LSTM baseline。GRDC 实测站点上同样最优（R² 0.5057、KGE 0.6612、F1 0.2427）[^src-rivermamba]。

## 相关页面

- [[mamba|Mamba]] — 选择性状态空间模型
- [[deep-state-space-model|深度状态空间模型]] — SSM 在深度学习中的发展
- [[space-filling-curves|空间填充曲线]] — 序列化/反序列化扫描路径
- [[location-aware-adaptive-normalization|LOAN]] — 位置感知自适应归一化
- [[flood-forecasting|洪水预报]] — AI 驱动的洪水预报领域
- [[s-mamba|S-Mamba]] — 首个 Mamba-based MTSF baseline
- [[dst-mamba|DST-Mamba]] — 分解式时空 Mamba
- [[stg-mamba|STG-Mamba]] — 首个 Mamba STG 预测

[^src-rivermamba]: [[source-rivermamba]]
