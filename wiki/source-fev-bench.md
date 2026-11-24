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
last_updated: 2026-11-24
source_count: 1
confidence: medium
status: active
---

# fev-bench: A Realistic Benchmark for Time Series Forecasting

- **来源**：Shchur\*、Ansari\*（Amazon），Turkmen（Keystone AI），Stella、Erickson（Prior Labs），Guerron（Amazon / Boston College），Bohlke-Schneider、Wang（NTT DATA AIVista）；核心切片未带会议水印、引文含 2026 年工作（本仓库 OpenReview CCF-A 调研记录其录用于 NeurIPS 2026 E&D）[^src-fev-bench]。
- **核心问题与基准构造**：现有预测基准领域偏窄、常剥离真实部署中常见的外生协变量（如 M5、Rossmann），且聚合排名缺少统计检验，难以区分模型真实增益与抽样噪声[^src-fev-bench]。论文构建覆盖 7 域、96 个唯一数据集共 100 项任务的 **[[fev-bench]]**（滚动窗口按序列数取 20/10/1，另含 20 任务的轻量版 `fev-bench-mini`），纳入 46 项含协变量任务（30 known dynamic、24 past dynamic、19 static）与 35 项多变量任务，并在 42 项动态协变量任务上设置协变量开关受控消融[^src-fev-bench]。评测以 MASE 与 9 分位点 SQL（$Q=\{0.1,\dots,0.9\}$，按季节朴素基线归一化）为主指标、WQL/WAPE 为辅；汇总边际/成对胜率 $W$ 与截断于 $[10^{-2}, 100]$ 的相对误差几何平均技能分 $S$，并执行 $B=1000$ 次任务级配对 Bootstrap 估计 95% 置信区间；自报训练重叠泄露统一以 `Chronos-Bolt (Base)` 插补，崩溃任务以 `Seasonal Naive` 填充，单卡 A10G 24 GB 显存受控推理[^src-fev-bench]。相对 GIFT-Eval 恢复被丢弃协变量并引入 SQL、配对 Bootstrap CI 与泄露插补；相对 [[timesx|TimesX]] 的严格时间截断防泄露路线，则采用自报重叠加基线分数插补[^src-fev-bench]。

## 证据

1. **主结论**：Chronos-2 双指标第一，TiRex 与 TimesFM-2.5 95% CI 重叠（第二名不可分）；重加权或去泄露后排序稳[^src-fev-bench]。
2. **归因**：42 项动态协变量开关将 Chronos-2/TabPFN-TS 提至 $47.0/42.5$（原 $40.9/34.1$）；35 项多变量微升（$54.4\to 55.0$ / Toto $54.1\to 54.7$）[^src-fev-bench]。
3. **缺口**：排亚分钟频，非盲测前瞻平台，域/频分布不均且限 24 GB[^src-fev-bench]。

## 相关页面

- [[fev-bench]] · [[probts]] · [[timesx]] · [[tsfm-covariate-adaptation-comparison]] · [[baguan-ts]] · [[chronos]] · [[tabpfn-ts]] · [[tirex]] · [[timesfm]]

[^src-fev-bench]: [[source-fev-bench]]
