---
title: "Flood Forecasting"
type: concept
tags:
  - hydrology
  - disaster-prediction
  - spatio-temporal
  - deep-learning
created: 2026-07-21
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

# Flood Forecasting（洪水预报）

洪水预报是指利用气象、水文和地理数据预测未来河流流量及洪水事件的任务。按类型可区分为：**河流洪水**（fluvial，河道溢流）、**海岸洪水**（coastal，风暴潮）和**暴雨洪水**（pluvial，极端降雨引发的内涝）[^src-rivermamba]。其中河流洪水预报是本 wiki 关注的核心方向。

## 从物理模型到深度学习

### 物理模型：GloFAS

全球洪水感知系统 GloFAS（ECMWF 运营）代表当前最先进的物理模型：用 LISFLOOD 水文模型驱动，输出全球 0.05° 日均流量再分析和集合预报。但物理模型运行昂贵、需要大量校准，且对复杂集水区特征的适应性有限[^src-rivermamba]。

### 深度学习路线

近年来深度学习被视作增强洪水预报的关键工具。主要路线演变[^src-rivermamba]：

1. **局部集总模型**（Lumped Models）：以 LSTM 为骨干（EA-LSTM、ED-LSTM、Hydra-LSTM、MC-LSTM、DRUM 等），假设单模型可跨集水区泛化，但**忽略空间拓扑和河流路由**。Google 全球运营系统即基于 Encoder-Decoder LSTM，仅在稀疏站点预报。
2. **图神经网络**（GNN）：尝试建模河网拓扑，但受限于小尺度，且多数 GNN 在捕获拓扑信息上失败。
3. **网格化+路由**：在粗网格上用 LSTM 估计产流，再以 1D 卷积或物理信息网络做河道路由。
4. **全局高分辨率深度学习**：RiverMamba（NeurIPS 2025）是首个突破——在 0.05° 全球网格上直接预测流量和洪水，使用 Mamba 选择性 SSM 实现线性复杂度时空建模。

## 洪水严重度度量

洪水严重度通过统计**重现期**（return period，年）来量化，范围为 1.5-500 年[^src-rivermamba]。高重现期（如 100 年一遇）反映统计稀有性，代表极端洪水事件。RiverMamba 将此信息直接编码到训练损失中——洪水事件按重现期加权，确保模型不会忽略罕见但高危的极端事件。

## RiverMamba 的突破

RiverMamba 首次实现[^src-rivermamba]：
- 全球 0.05° 高分辨率连续流量图（而非稀疏站点）
- 1-7 天中期预报，覆盖河流洪水全部重现期（包括极端事件）
- 显式时空建模：通过空间填充曲线序列化连接完整河网
- 超越 GloFAS 物理模型和 LSTM baseline 的预报精度

## 相关页面

- [[rivermamba|RiverMamba]] — 全球洪水预报 Mamba 模型
- [[precipitation-nowcasting|降水临近预报]] — 0-8h 短时降水预测
- [[extreme-weather-forecasting|极端天气预报]]
- [[spatio-temporal-foundation-model|时空基础模型]]

[^src-rivermamba]: [[source-rivermamba]]
