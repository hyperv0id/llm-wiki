---
title: "TimeBench"
type: entity
tags:
  - time-series
  - dataset
  - pretraining
  - icml-2025
  - tsinghua
created: 2026-06-08
last_updated: 2026-08-08
source_count: 1
confidence: high
status: active
---

# TimeBench

## 概述

**TimeBench** 是由 Sundial 团队 (Tsinghua University, BNRist) 为预训练时间序列基础模型而构建的**万亿级时间序列数据集**，包含 1.032 万亿 (1.032T) 时间点[^src-sundial]。

## 数据集构成

| 来源 | 时间点 | 占比 | 说明 |
|------|--------|------|------|
| LOTSA (Liu et al., 2024) | 230B | 22.29% | 大规模开源时序数据 |
| ERA5 Daily | 406B | 39.35% | 全球再分析日度气象数据 |
| ERA5 3h | 129B | 12.50% | ERA5 三小时气象数据 |
| Chronos (Ansari et al., 2024) | 94B | 9.11% | Chronos 团队收集的时序数据 |
| ERA5 Monthly | 58B | 5.62% | ERA5 月度气象数据 |
| ERA5 Weekly | 32B | 3.10% | ERA5 周度气象数据 |
| ECG (Goldberger et al., 2000) | 48B | 4.65% | 心电图生物信号 |
| Finance (自收集) | 10.5B | 1.02% | 金融领域真实时序 |
| ERA5 Quarterly | 13.5B | 1.31% | ERA5 季度气象数据 |
| IoT (自收集) | 5.8B | 0.56% | 物联网传感器数据 |
| ERA5 12h | 4.5B | 0.44% | ERA5 12 小时气象数据 |
| Synthetic (合成) | 0.5B | 0.05% | 用于提升模式多样性的合成数据 |

## 数据预处理

- 缺失值插补
- 异常值排除  
- 归一化技术
- 统计特性分析（非平稳性、可预测性、季节性）
- 合成技术增强模式多样性

## 特性

- **多领域**: 金融、物联网、气象学、医疗健康
- **多频率**: 3小时、12小时、每日、每周、每月、季度
- **多长度和多变元数**: 覆盖全面的时序动态和变化模式
- **数据泄露防护**: 所有用于评估的数据集（TSLib, GIFT-Eval, FEV）均被排除，确保真正零样本预测

## 重要性

TimeBench 是目前**最大规模**的时间序列预训练数据集，验证了时间序列基础模型的缩放定律——更大的预训练数据持续带来更好的零样本性能（Table 8: Sundial 94B → 230B → 1032B）[^src-sundial]。

## 相关页面

- [[sundial]] — 基于 TimeBench 预训练的 Sundial 模型系列
- [[timeflow-loss]] — Sundial 使用的生成式训练目标
- [[chronos]] — Chronos，使用 94B 时间点子集
- [[time-300b]] — 同期发布的 309B 时间点开源时序数据集

[^src-sundial]: [[source-sundial]]
