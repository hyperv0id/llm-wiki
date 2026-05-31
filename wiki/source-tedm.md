---
title: "TEDM: Time Series Forecasting with Elucidated Diffusion Models"
type: source-summary
tags:
  - diffusion-models
  - time-series-forecasting
  - edm
  - score-based-sde
  - iclr-2026
  - multivariate-forecasting
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: medium
status: active
---

# TEDM: Time Series Forecasting with Elucidated Diffusion Models

**Solano-Carrillo, Naveenachandran & Niebling (DLR / German Aerospace Center), ICLR 2026**

## 核心论题

TEDM 是首个将 EDM 的完整设计空间阐明框架从图像迁移到时间序列预测的工作[^src-tedm]。核心贡献有两条：(1) 将扩散时间轴与物理时间轴对齐，将采样复杂度从 O(SH) 降到 O(H)；(2) 首次使用从数据中经验估计的噪声 (Σt) 和尺度 (st) schedule，避免人工预设 schedule 引入的归纳偏置[^src-tedm]。

## 方法

### 扩散-物理时间轴对齐
EDM 的 ODE 求解需要在每个扩散时间步内对每个物理时间步多次评估网络。TEDM 注意到其导出的差分方程不显式依赖 dt，因此将扩散时间轴等同于物理时间轴，每个 Euler 步直接推进一个预测步[^src-tedm]。从给定窗口 y1:T 开始，H 个 Euler 步后直接获得预测窗口 yT+1:T+H，总推理时间 O(H) 而非 O(SH)[^src-tedm]。

### 数据驱动的 schedule
传统扩散模型使用预设的噪声/尺度 schedule（如 EDM 的 σt=t, st=1）。TEDM 从理论上推导 E(xt)=st·E(x0) 和 Cov(xt)=s²t·Σt，从而从输入窗口 y1:T 经验估计 st 和 Σt[^src-tedm]。提供两种估计方案：累积估计（cumulative）和滑动窗口估计（sliding window），后者对局部统计变化更灵活[^src-tedm]。

### 结构化噪声与预处理
噪声不再是 i.i.d. 高斯，而是通过 Σt^(1/2) 施加结构化噪声，每个时间步和特征维度可以有不同的噪声水平[^src-tedm]。EDM 的预处理方案被推广到矩阵值 Σ，cskip/cout/cin 等预处理项以 Cov(y) 和 Σ 的矩阵形式表达，在对角近似下退化为逐维度的 EDM 预处理[^src-tedm]。

### 推断
训练后，使用 Euler 步沿 ODE（或 SDE）积分进行预测。在对角 Σt 近似下，推断简化为逐元素的运算，效率极高[^src-tedm]。

## 实验结果

- 6 数据集上对比 5 个扩散 baselines（TimeDiff, DiffusionTS, TMDM, ARMD, NsDiff），TEDM 在 ETTh2（MSE 0.214）、ETTm2（MSE 0.135）、Exchange（MSE 0.069）上取得 SOTA[^src-tedm]
- 消融实验：TEDM 相比 EDM 最高提升 85% MSE（66% MAE）；经验 schedule 比 st=1 持续更好；滑动窗口估计通常优于累积估计[^src-tedm]
- 资源效率：TEDM 训练每 batch 仅 0.004s（ARMD 0.009s），推理 0.11s，训练内存 21.3 MB[^src-tedm]
- 相比非扩散方法（iTransformer, TimesNet, DLinear, PatchTST），TEDM 在多数数据集上也占优[^src-tedm]

## 局限性

- 基于 Ito 扩散的假设不适用于长记忆过程（fractional Brownian motion）、重尾噪声（α-stable）和跳过程[^src-tedm]
- 对高维特征空间（如 Solar 的 137 维）效果下降，因为对角 Σt 近似可能不适用[^src-tedm]
- ETTh1 上的大振幅变化场景表现较弱，违背光滑流假设[^src-tedm]

[^src-tedm]: [[source-tedm]]
