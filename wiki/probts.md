---
title: "ProbTS"
type: entity
tags:
  - time-series
  - benchmark
  - probabilistic-forecasting
  - point-forecasting
  - foundation-model
  - neurips-2024
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: medium
status: active
---

# ProbTS

**ProbTS** 是 Microsoft Research Asia 与 HKUST(GZ)/清华联合提出的时间序列预测基准工具，发表于 NeurIPS 2024 Datasets and Benchmarks Track（arXiv:2310.07446）[^src-probts]。它把“跨 horizon 的精确点预测 + 可靠分布预测”视为统一本质需求，提供数据集、指标、模型与分析管线，并开源于 `microsoft/ProbTS`[^src-probts]。

## 设计目标

既有工作常分别优化长程点预测或短程概率估计，导致方法偏好被局部场景扭曲[^src-probts]。ProbTS 的目标是：

1. 在**短程与长程**设定下同时评估模型；
2. 同时报告**点指标**（NMAE、MAE、MSE、NRMSE、MASE 等）与**分布指标**（CRPS、CRPS-sum）；
3. 用可量化数据特征解释“为何某类方法只在某类场景赢”；
4. 显式对比 [[ar-vs-nar-decoding|AR vs NAR]]、分布估计方式与归一化策略[^src-probts]。

## 核心模块

| 模块 | 内容 |
|------|------|
| Data | 短程：Exchange/Solar/Electricity/Traffic/Wikipedia；长程：ETT/Electricity/Traffic/Weather/Exchange/ILI 等；统一切分与预处理[^src-probts] |
| 数据特征 | 趋势 $F_T$、季节 $F_S$（STL 分解）、[[non-gaussianity|非高斯性]]（窗口内 JS(经验分布 ‖ 拟合高斯)）[^src-probts] |
| Model | 编码器 $f_\phi$ + 预测头 $p_\theta$；AR：逐步 $h_t=f_\phi(x_{t-1},c_t,h_{t-1})$；NAR：一次生成 $h_{t+1:t+T}$[^src-probts] |
| 分布估计 | 点头 / 预定义似然头（Gaussian、Student-t、混合等）/ 神经生成模块（RealNVP、MAF、Diffusion）[^src-probts] |
| 归一化 | [[instance-normalization|RevIN]]（长程点预测线常见）vs 均值缩放（短程概率线常见）[^src-probts] |
| Evaluator | 统一反归一化后的点/分布指标；概率采样常用 100 条轨迹估计经验 CDF[^src-probts] |

## 覆盖模型（代表性）

- **长程点预测**：iTransformer、[[patchtst|PatchTST]]、TimesNet、N-HiTS、LTSF-Linear 等[^src-probts]
- **短程/概率**：GRU-NVP / GRU-MAF / Trans-MAF、[[timegrad|TimeGrad]]、[[csdi|CSDI]][^src-probts]
- **通用骨架与朴素基线**：Linear、GRU、Transformer、global/batch mean[^src-probts]
- **基础模型（零样本）**：Lag-Llama、[[timesfm|TimesFM]]、Timer、MOIRAI、[[chronos|Chronos]]、UniTS 等[^src-probts]

## 关键经验结论

- 长程定制架构的优势在短程、高非高斯性场景显著收缩[^src-probts]。
- 既有概率方法在长程分布预测上整体不足：AR 误差累积，NAR 扩散显存/效率受限[^src-probts]。
- AR 在强季节性上可反超 NAR 点预测模型；RevIN 常缓解长程 AR 误差累积，但短程概率线更依赖均值缩放[^src-probts]。
- TSFM 复现同样的 AR 长程劣势与复杂分布建模短板；领域专用概率模型（如 CSDI）在高非高斯性上仍不可被预定义分布头轻易替代[^src-probts]。

## 相关页面

- [[source-probts]] — 源摘要
- [[ar-vs-nar-decoding]] — 解码方案轴
- [[non-gaussianity]] — 分布复杂度指标
- [[instance-normalization]] — RevIN
- [[generative-time-series-forecasting]] — 生成式预测范式
- [[timegrad]] / [[csdi]] / [[patchtst]] / [[timesfm]] / [[chronos]]

[^src-probts]: [[source-probts]]
