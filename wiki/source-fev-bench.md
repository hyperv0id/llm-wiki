---
title: "fev-bench: A Realistic Benchmark for Time Series Forecasting"
type: source-summary
tags:
  - time-series-forecasting
  - benchmark
  - evaluation
  - covariate-aware-forecasting
  - foundation-model
  - neurips-2026
created: 2026-11-24
last_updated: 2026-11-27
source_count: 1
confidence: medium
status: active
---

# fev-bench: A Realistic Benchmark for Time Series Forecasting

现有通用预测基准常在清洗时丢掉促销、价格、日历等外生协变量，且跨任务聚合只报平均名次、缺少显著性检验，预训练泄露处理口径也不统一。论文提出 [[fev-bench]]，在 7 个领域、96 个公开数据集构成的 100 项任务上系统纳入三类协变量，并配以配对置信区间与统一泄露插补协议；核心实证发现，现有时间序列基础模型的性能差异主要来自是否利用外生协变量，而非原生多变量交互[^src-fev-bench]。

## 核心做法

- **协变量与受控消融**：涵盖已知动态、历史动态与静态三类协变量，并在动态协变量任务与多变量任务上分别做开关消融，以解耦骨干能力、协变量增益与多变量交互；另提供 20 项任务的轻量子集 `fev-bench-mini`[^src-fev-bench]。
- **尺度无关指标与统计聚合**：以 MASE 与按季节朴素归一化的 9 分位点 SQL 为主指标，聚合平均胜率与截断相对误差几何平均技能分，用任务级配对 Bootstrap 估计 95% 置信区间区分真实差距与抽样噪声[^src-fev-bench]。
- **自报泄露插补**：对模型方自报与预训练语料重叠的任务不直接剔除，而是统一用全零样本单变量基线 `Chronos-Bolt (Base)` 的分数插补（崩溃任务以季节朴素填充），使所有模型仍在同一任务集上可比且基准不缩水[^src-fev-bench]。

## 证据

1. **定位与主结论**：Chronos-2 在双主指标居首，预训练模型整体强于统计基线，而 TiRex 与 TimesFM-2.5 置信区间重叠、第二名统计不可分[^src-fev-bench]。
2. **真实涨点归因**：喂入动态协变量使 Chronos-2 与 [[tabpfn-ts|TabPFN-TS]] 技能分大幅跃升，而开启原生多变量模式仅带来边际改善，增益主源是外生协变量[^src-fev-bench]。
3. **证据缺口**：排除亚分钟高频、属静态公开集而非前瞻盲测（泄露依赖自报），且各领域与频率的任务分布不均[^src-fev-bench]。

## 相关页面

- 瓶颈接力 $\Delta$ → [[probts]]：把统一点/概率评估推进到现实外生协变量任务，并补齐配对置信区间与泄露插补。
- 同槽位互斥 → [[timesx]]：同为防预训练泄露评测，前者用严格时间截断隔离，本文用自报加基线插补。
- 跨槽位正交 → [[chronos]]、[[tabpfn-ts]]：协变量开关消融中受益最明显的被测模型，基准协议与模型能力互为验证。
- 复现对齐 → [[baguan-ts]]：复用 fev-bench 含协变量子集对其原生协变量 ICL 能力做同台验收。

出处：Shchur 与 Ansari 共同一作（Amazon）等 8 人，NeurIPS 2026 Evaluations and Datasets track。

[^src-fev-bench]: [[source-fev-bench]]
