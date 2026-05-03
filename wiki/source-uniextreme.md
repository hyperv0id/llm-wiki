---
title: "UniExtreme: A Universal Foundation Model for Extreme Weather Forecasting"
type: source-summary
tags:
  - weather-forecasting
  - foundation-model
  - extreme-weather
  - frequency-domain
  - memory-network
  - arxiv-2025
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# UniExtreme: A Universal Foundation Model for Extreme Weather Forecasting

**Authors**: Hang Ni, Weijia Zhang, Hao Liu (HKUST Guangzhou). **arXiv**: 2508.01426v2, September 2025[^src-uniextreme].

## 核心论点

现有天气基础模型（GraphCast、PanguWeather 等）专注于一般天气条件，对极端天气事件的预测能力有限。此前极端天气方法仅处理单一事件类型（如极端降水、热浪），无法泛化到多样化的极端现象。UniExtreme 是首个同时利用多样化真实极端天气事件标注数据和一般气象数据训练的基础模型，可在统一模型下预测 18 种极端天气事件而无需额外微调[^src-uniextreme]。

## 两个关键发现

1. **频谱差异**：极端天气区域的高频面积（HFA）分布相比正常天气呈显著"右移"，Wasserstein 距离为 3.1e-3（正常-极端）vs 2.4e-4（正常-随机），表明极端天气能量更集中于高频分量[^src-uniextreme]。
2. **层次化驱动与地理混合**：极端事件具有层次结构——不同事件类型由不同物理驱动因素支配（类间差异），同一类型内也有多种大气模式（类内差异）。约 86% 的时间步存在复合极端事件，平均点级重叠率 69%[^src-uniextreme]。

## 两个核心模块

1. **自适应频率调制（AFM）**：多组可学习 Beta 分布滤波器对不同频段进行区域自适应调制，加上时空感知的频段聚合网络，区分正常与极端天气频谱[^src-uniextreme]。
2. **事件先验增强（EPA）**：从训练数据中构建分类极端事件记忆池，通过双层（类内+类间）注意力融合网络捕获层次化模式和复合效应[^src-uniextreme]。

## 实验结果

在 HR-Extreme-V2 数据集（26TB，2019-2024，美国本土 6km 分辨率）上，UniExtreme 在极端天气预测上取得约 11% MAE 和 10% RMSE 的改进（相比最佳基线），同时在一般天气预测上也表现更优。在 MSL 变量上，将正常-极端性能差距缩小约 37%[^src-uniextreme]。

## 局限性

- 仅评估 1 小时临近预报（nowcasting），未验证中长期预测
- 频率分析基于 2D RFFT，未考虑时间维度的频谱特性
- EPA 记忆仅使用 2022 年极端事件记录，容量固定为 U=5
- Dust Devil（DstD）事件在 ACC 指标上被部分基线超越

[^src-uniextreme]: [[source-uniextreme]]
