---
title: "AliO: Output Alignment Matters in Long-Term Time Series Forecasting"
type: source-summary
tags:
  - time-series-forecasting
  - long-term-forecasting
  - direct-forecast
  - consistency-regularization
  - frequency-domain
  - neurips-2025
created: 2026-11-24
last_updated: 2026-11-27
source_count: 1
confidence: medium
status: active
---
# AliO: Output Alignment Matters in Long-Term Time Series Forecasting

直接多步预测一次前向输出整段未来，但历史窗口只平移一格，两路预测在重叠的时间点上就常互相矛盾：同一段曲线被整体推移。这种不一致不进入 MSE，却让按预测调配资源的人反复改计划。AliO 在训练期让两个相邻窗口一起前向，对重叠片段施加时域与复频域一致性约束，并把距真值更远的一侧单向拉向更近的一侧；不改骨干、不加推理开销，不一致度大幅下降而回归误差多数持平[^src-alio]。

## 核心做法

- **双前向**：错开一步的两段历史各前向一次，截出落在同一批未来点上的重叠片段。
- **时频域对齐**：重叠片段在时域逐点取均方距离，再各做一次 `FFT`、在复频谱上取距离。时域项压波形偏移，频域项补幅度与相位——只压时域时，峰谷错位而形状看似相同。
- **单向拉齐**：逐点比较两路与真值的误差，用 stop-gradient 冻结距真值更近的一侧，仅让更远的一侧回传梯度并向更近侧拉齐；对称回传会让两路收敛到折中的错误均值。
- **开销**：训练每步两次前向加一次重叠窗 `FFT`，推理沿用单次前向[^src-alio]。

## 证据

1. **定位与主结论**：跨多种骨干，相邻窗口的不一致度普遍大幅下降，回归误差多数持平，仅原本一致性极差的 ILI（流感样病例基准）降幅明显[^src-alio]。
2. **真实涨点归因**：只做时域与频域权重的敏感性分析，缺单独去掉单向梯度截断或频域项的组件消融[^src-alio]。
3. **证据缺口与负结果**：收益集中在初始一致性差的数据，高波动数据集收益有限甚至可能推高误差[^src-alio]。

## 相关页面

- [[direct-forecast|Direct Forecast（DF）]] — 瓶颈接力 $\Delta$：DF 一次输出整段未来从而暴露跨窗口重叠不一致瓶颈，AliO 在该前提下接力对齐。
- [[distdf|DistDF]] — 同槽位互斥：它对齐未来各步条件分布矩，AliO 对齐跨窗口预测。
- [[qdf|QDF]] — 同槽位互斥：它学协方差型加权损失校正自相关偏差，AliO 压相邻窗口分歧。
- [[fredf|FreDF]] — 跨槽位正交：同用可微频域损失，它对「预测↔标签」，AliO 对「预测↔预测」。

NeurIPS 2025（UNIST）[^src-alio]

[^src-alio]: [[source-alio]]
