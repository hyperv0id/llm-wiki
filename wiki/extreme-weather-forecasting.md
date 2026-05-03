---
title: "Extreme Weather Forecasting"
type: concept
tags:
  - weather-forecasting
  - extreme-weather
  - foundation-model
  - frequency-domain
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# Extreme Weather Forecasting

极端天气预测（Extreme Weather Forecasting）是指对罕见、非线性、高度敏感的大气极端事件进行预报的任务，包括洪涝、龙卷风、冰雹、闪电、热浪、寒潮等多种现象[^src-uniextreme]。

## 核心挑战

### 1. 频谱差异

极端天气区域在频域上与正常天气存在系统性差异。UniExtreme 对 2024 年美国约 3640 万正常区域和 88.2 万极端区域的高频面积（HFA）分析显示，极端区域的 HFA 分布呈显著"右移"——Wasserstein 距离为 3.1e-3（正常-极端）vs 2.4e-4（正常-随机），表明极端天气能量更集中于高频分量[^src-uniextreme]。这一发现跨越 2019-2024 全部年份，具有广泛普遍性。

> [!note] 与现有频域方法的关系
> OneForecast 使用高通 GNN 分离高频信号，但抑制了其他频段的有用信息[^src-uniextreme]。[[adaptive-frequency-modulation|AFM]] 的 Beta 滤波方法通过自适应多频段调制避免了这一缺陷。

### 2. 层次化驱动与地理混合

极端事件呈现层次结构：不同事件类型由不同物理驱动因素支配（类间差异），同一类型内有多种大气模式（类内差异）[^src-uniextreme]。此外，极端事件往往同时发生——2024 年美国约 86% 的时间步存在复合极端事件，平均点级重叠率 69%。例如美国东海岸可能同时经历暴雨、洪涝和强风。

### 3. 数据稀缺性

极端事件在时间序列中占比极低，导致模型训练时样本不均衡，容易对极端模式过平滑[^src-uniextreme]。

## 方法演进

### 单一事件类型方法

早期方法针对特定极端现象设计：NowcastNet（极端降水，Nature 2023）、FuXiExtreme（极端降雨和风力，Science China 2024）、López-Gómez et al.（热浪，AI for the Earth Systems 2023）。这些方法无法泛化到其他极端事件类型[^src-uniextreme]。

### 广义极端特征方法

- **ExtremeCast**：通过极值理论（EVT）引导的损失函数缓解过平滑，但仅基于百分位事件评估[^src-uniextreme]
- **OneForecast**：使用高通 GNN 捕获极端模式，但缺乏多样化真实极端事件的监督信号[^src-uniextreme]

### 通用极端天气基础模型

**[[uniextreme|UniExtreme]]** 是首个利用多样化真实极端天气标注数据的基础模型，覆盖 18 种事件类型。通过 [[adaptive-frequency-modulation|AFM]] 解决频谱差异，通过 [[event-prior-augmentation|EPA]] 处理层次化和复合极端[^src-uniextreme]。

## 与其他预测领域的关系

- [[traffic-forecasting|交通预测]]中的事故感知方法（如 [[conformer|ConFormer]]、[[igstgnn|IGSTGNN]]）同样面对罕见事件建模问题，但领域和时序特性不同
- [[multimodal-time-series-forecasting|多模态时间序列预测]]关注文本/图像辅助，而极端天气预测关注频域和事件先验
- [[generative-time-series-forecasting|生成式时间序列预测]]关注不确定性量化，极端天气预测当前以判别式方法为主

[^src-uniextreme]: [[source-uniextreme]]
