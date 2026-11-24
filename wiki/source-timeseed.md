---
title: "TimeSeed: Effective Time Series Forecasting with Sparse Endogenous Variables"
type: source-summary
tags:
  - time-series-forecasting
  - exogenous
  - sparse-endogenous
  - frequency-domain
  - linear-models
  - icml-2026
created: 2026-11-24
last_updated: 2026-11-24
source_count: 1
confidence: medium
status: active
---

# TimeSeed: Effective Time Series Forecasting with Sparse Endogenous Variables

**来源**：Zhaowang Wu、Kaixin Deng、Hua Yan（四川大学电子信息学院；北海道大学信息科学研究科）；ICML 2026（PMLR 306）[^src-timeseed]。

现有基于外生变量的时序预测模型默认内生历史观测完整；当内生变量极度稀疏（仅剩 4% 观测点甚至单点）时，外生向内生的直接映射会出现严重的尺度与因果失配，导致强基线普遍退化。为此，TimeSeed 将「直接从极稀疏样本外推未来」改写为「先借助稠密外生序列重建完整的内生历史上下文，再执行常规长时预测」[^src-timeseed]。

## 去黑话算子配方与路线差异

模型完全由标准算子组合而成，未引入改变复杂度阶数的新底层算子[^src-timeseed]：

- **时序分解**：用 `MovingAvg` 将稠密外生序列拆分为趋势项与周期项[^src-timeseed]。
- **时域与频域聚合**：时域分支对周期项切 patch 后依次沿变量轴与周期轴做 Linear 交互；频域分支对趋势项做 `RFFT`，按幅值能量划分为高能（Top-$K$）与低能两组，各沿变量轴接频域 Linear 压缩为单变量频谱，相加后经 `iRFFT` 还原趋势上下文[^src-timeseed]。
- **多尺度历史重建与预测**：对极稀疏内生序列 $S$ 分别上采样 $2^q$（$q=1..3$）并过 MLP 专家，经 `Dense + Gumbel-Softmax` 软/硬选出单一最优分辨率重建历史，与时频外生上下文相加后再过时间轴 Linear 输出预测[^src-timeseed]。
- **有向关系对照**：相对 [[source-timexer|TimeXer]] 的 patch-to-variate 交叉注意力与 [[crosslinear|CrossLinear]] 的 1D 卷积嵌入，TimeSeed 属**同槽位互斥与场景分支**，以沿变量轴的纯线性聚合换取 10.5 MB 显存（vs TimeXer 14.69 MB）、约 30% 训练加速与约 0.19M 参数量，代价是放弃变量的注意力自适应选择；相对 [[timemixer|TimeMixer]] 的 `AvgPool` 下采样多预测器求和，TimeSeed 构成**同构反转**（改为多倍率上采样 + Gumbel 单尺度选择）；相对 [[direct-forecast|Direct Forecast]] 与 [[tide|TiDE]] 则形成输入侧两阶段历史重构对直接多步映射的范式对照[^src-timeseed]。

## 证据

1. **定位与主结论**：4%/单点稀疏下，7 项 LTSF 与 PhysioNet 整体优于 11 基线及两步插补管线，10.5 MB 显存与速度占优[^src-timeseed]。
2. **真实涨点归因**：单模块移除使 MSE 恶化 5.31%–12.30%，但频域高能分带最弱（5.31%）；主涨点来自历史重构改写直接预测（ETTh2/ETTm2 降 MSE 27.4%、MAE 14.2%），Gumbel 负责匹配局部尺度[^src-timeseed]。
3. **证据缺口与负结果**：ETTm1 上两步「插补+PatchTST」（0.041）胜过 TimeSeed（0.068）；漏测 TSFM；对外生数量/长度敏感（减外生掉 15.32%），零样本冷启动待探索[^src-timeseed]。

## 相关页面

- [[source-timexer]] — 稠密内生下的注意力外生融合基线（同槽位互斥与场景分支）
- [[crosslinear]] — 1D 卷积外生交叉相关线性基线（同槽位互斥）
- [[timemixer]] — AvgPool 下采样多尺度混合（多尺度方向与选择同构反转）
- [[tide]] — 残差 MLP 协变量直接预测（同槽位互斥）
- [[direct-forecast]] — 直接多步预测范式（两阶段历史重构对照）
- [[tsfm-covariate-adaptation-comparison]] — 协变量适配全景矩阵（输入侧历史重构定位与漏测 TSFM 缺口）

[^src-timeseed]: [[source-timeseed]]
