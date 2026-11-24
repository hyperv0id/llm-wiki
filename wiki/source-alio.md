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
last_updated: 2026-11-24
source_count: 1
confidence: medium
status: active
---

# AliO: Output Alignment Matters in Long-Term Time Series Forecasting

- **作者/机构**：Kwangryeol Park、Jaeho Kim（UNIST / Korea Univ.）；Seulki Lee（UNIST）[^src-alio]
- **出处**：NeurIPS 2025[^src-alio]
- **分级**：Tier 2（轻量化入库，未为 `alio` 或 `output-alignment` 新建 `source_count: 1` 独立页）

直接多步预测（DF）模型对仅错开一个时间步的两个相邻历史窗口分别前向预测时，对重叠的未来时间点往往给出互相矛盾的数值。AliO 不修改任何预测骨干结构，仅在训练期同步前向两个相邻滚动窗口，对未来重叠片段同时施加时域与频域一致性对齐损失，并借助单向梯度截断（stop-gradient Regression Pulling）将距真值更远的一侧拉向更近的一侧，避免两路预测一起向错误均值坍缩[^src-alio]。

## 去黑话算子配方与路线差异

模型未引入改变推理复杂度的新网络算子，仅在训练损失侧附加跨窗口时频对齐与单向门控[^src-alio]：

- **滑窗双前向与重叠对齐**：取错开步长 $l$（默认 $l=1, N=2$）的相邻历史窗口各前向一次，截取未来重叠片段，在时域计算 MSE/MAE 一致性损失 $\mathcal{L}_T$，并经 `FFT` 转换后计算复频域一致性损失 $\mathcal{L}_F$（ComplexMSE）[^src-alio]。
- **单向 stop-gradient Regression Pulling**：逐点比较两路重叠输出与真实标签的误差距离，通过索引门控固定仅回传距离真值更远那一侧的梯度；总目标为 $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda_T\mathcal{L}_T + \lambda_F\mathcal{L}_F$[^src-alio]。训练期每步需双倍前向传播与重叠窗 `FFT`，推理阶段保持单次前向、零额外开销[^src-alio]。
- **有向关系对照**：归属 [[direct-forecast|Direct Forecast（DF）]] 前提设定（仅一次输出整段未来步才存在跨窗口重叠预测不一致）；相对 [[distdf|DistDF]] 与 [[qdf|QDF]] 属**同槽位互斥**（均不改骨干且推理零开销，但 AliO 对齐跨窗口预测间一致性，DistDF/QDF 对齐条件分布矩或可学习协方差二次型似然）；相对 [[fredf|FreDF]] 属**跨槽位正交**（同用可微频域损失但对齐对象不同：AliO 对齐跨窗口「预测↔预测」重叠复频谱，FreDF 对齐单窗口「预测↔标签」频域幅度）；对 [[patchtst|PatchTST]] 属**实证复现对齐**（沿用官方 $L=336$ 回看配置）[^src-alio]。

## 证据

1. **定位与主结论**：属「滚窗重叠一致性正则化」——跨 7 骨干显著降 TAM（表 1 最高 70.5%），常规 MSE 在 ECL/Weather 基本持平，大幅降错集中于初始一致性极差的 ILI（降 17.4%）[^src-alio]。
2. **消融缺口**：正文仅测 $\lambda_T, \lambda_F$ 与 $(l, N)$ 敏感性（$\lambda_T$ 增至 5 时 TAM 与 MSE 同向改善），**缺「去 Regression Pulling / 去频域对齐」独立组件消融表**[^src-alio]。
3. **口径与代价**：摘要（TAM 降 58.2%、MSE 最高降 27.5%）与正文表 1（70.5%、11.5%/ILI 17.4%）口径不一；训练需双前向与重叠窗 `FFT`；高波动集慎用[^src-alio]。

## 相关页面

- [[direct-forecast]] — 前提归属（一次输出整段未来步才存在跨窗口重叠预测不一致）
- [[distdf]] · [[qdf]] — 同槽位互斥（即插即用训练目标：跨窗口输出一致性 vs 条件分布/协方差似然对齐）
- [[fredf]] — 跨槽位正交（频域损失对齐对象：跨窗口预测↔预测复频谱 vs 单窗口预测↔标签幅度）
- [[patchtst]] — 实证复现对齐（沿用官方 $L=336$ 回看配置）

[^src-alio]: [[source-alio]]
