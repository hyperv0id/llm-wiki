---
title: "Time-Aligned Retrieval Enhancement"
type: technique
tags:
  - retrieval-augmented
  - time-alignment
  - temporal-regularity
  - time-series-forecasting
created: 2026-08-19
last_updated: 2026-08-19
source_count: 1
confidence: medium
status: active
---

# Time-Aligned Retrieval Enhancement

时间对齐检索增强（Time-Aligned Retrieval Enhancement）是一种在形态相似度之上叠加时间对齐奖励的检索技术，旨在抑制"形态相似但时间错位"的候选段，提升检索证据的可靠性[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

## 动机

时间序列常展现日历驱动的规律性（如日内、周内、季节性模式）。纯形态检索可能返回形态相似但时间上下文错位的段——例如，查询属于周二上午，但检索到的段来自周六同一时刻——其对应未来可能因周期错位而失配[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

## 机制

给定查询时间戳 {t_q} 和训练时间戳 {t_i}，计算时间对齐奖励矩阵 B ∈ R^{B×N}[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]：

$$B_{q,i} = \sum_{r \in R} \lambda_r \phi_r(t_q, t_i)$$

其中 R 包含 hour-of-day、day-of-week、month-of-year 和 minute-of-hour（当可用时）。对于循环分量（小时、月、分），使用圆形距离计算差异以处理回绕（如 23 和 0 接近）。对于周分量，精确工作日匹配获得最高奖励，同类型（工作日 vs 周末）获得较小奖励[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

每个分量 φ_r 使用指数核实现，B 最后按最大值归一化到 [0,1][^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

## 与相似度的融合

最终相似度为形态相似度与时间对齐奖励的加权组合[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]：

$$S_{sim}(q,i) = (1 - \alpha_{time}) S_{temporal}(q,i) + \alpha_{time} B_{q,i}$$

其中 S_temporal 为 Pearson 相关相似度，α_time 控制时间对齐的贡献权重。

## 实验证据

在 [[saraf|SARAF]] 的消融实验中，时间对齐增强是贡献最一致的组件：移除后所有数据集 MSE 上升[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。其效果具有数据集依赖性：在较非平稳的 ETTh2 上 α_time 影响更明显，长预测步（H=720）时尤甚；在较平稳的 Electricity 上影响较小[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

## 与其他方法的关系

- 与 [[gtr|GTR]] 的周期信息对齐：GTR 通过可学习参数矩阵按绝对时间位置检索周期模式，时间对齐增强则在检索阶段通过奖励矩阵隐式引导时间对齐[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。
- 与 [[cyclenet|CycleNet]] 的周期建模：CycleNet 通过残差周期建模显式编码周期性，时间对齐增强在检索阶段利用周期性[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]: [[source-stationarity-aware-retrieval-augmented-forecasting-kdd26]]
