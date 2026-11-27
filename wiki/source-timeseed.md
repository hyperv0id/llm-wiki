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
last_updated: 2026-11-27
source_count: 1
confidence: medium
status: active
---

# TimeSeed: Effective Time Series Forecasting with Sparse Endogenous Variables

长时预测与既有外生融合方法默认目标变量（内生变量）历史完整，当内生序列稀疏到约 4% 观测点甚至单点时，从外生直接映射未来会出现尺度与变量耦合错配而明显退化。TimeSeed 把「稀疏直接外推」改写成两阶段：先用稠密外生序列重建完整的内生历史上下文，再做常规长时预测；论文报告其在极端稀疏设定下整体优于十一类基线与「先插补再预测」管线[^src-timeseed]。

## 核心做法

稠密外生序列先经移动平均分解为周期项与趋势项，配合极稀疏内生观测重建历史上下文后，经沿时间轴的线性头直接预测未来[^src-timeseed]：

- **分块线性时域聚合**（Time Domain Aggregator）：周期项切 patch 后先沿变量轴线性聚合、再做跨周期线性交互[^src-timeseed]。
- **高低能分组频域聚合**（Frequency Domain Aggregator）：趋势项做实数 FFT，按能量分高低两组分别沿变量轴频域线性压缩后逆变换合并[^src-timeseed]。
- **多倍率上采样单选重建**（Adaptive Scale Reconstructor）：将极稀疏内生观测按多倍率上采样送入小 MLP 专家，以 Gumbel-Softmax 选出与观测密度匹配的单一尺度表征[^src-timeseed]。

## 证据

1. **定位与主结论**：极端稀疏下七个长时基准与 PhysioNet 整体最优，平均 MSE 降约一到两成，纯线性开销显著低于注意力基线[^src-timeseed]。
2. **真实涨点归因**：两阶段历史重建范式本身贡献最大（较直接预测降约四分之一），移除单模块仅恶化约半成到一成、频域高能分带最小[^src-timeseed]。
3. **证据缺口**：ETTm1 上「插补+PatchTST」反占优；未对比时序基础模型；外生变少或历史变短明显掉点[^src-timeseed]。

## 局限

论文自述仍需至少少量内生观测点，内生完全缺失（零观测点）场景留作未来工作[^src-timeseed]。

## 相关页面

- [[source-timexer|TimeXer]] — 同槽位互斥：TimeXer 用注意力且默认内生完整，TimeSeed 以纯线性面向稀疏内生场景。
- [[timemixer|TimeMixer]] — 同构/特例：TimeMixer 多倍率下采样求和，TimeSeed 多倍率上采样后单选最优尺度。
- [[direct-forecast]] — 瓶颈接力 $\Delta$：历史重建前级用于消除直接多步映射在稀疏输入下的尺度失配瓶颈。
- [[tsfm-covariate-adaptation-comparison]] — 跨槽位正交：协变量适配全景矩阵中的「输入侧历史重构」支线。

出版信息：ICML 2026（PMLR 306）；Zhaowang Wu、Kaixin Deng、Hua Yan。

[^src-timeseed]: [[source-timeseed]]
