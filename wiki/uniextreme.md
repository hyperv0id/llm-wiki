---
title: "UniExtreme"
type: entity
tags:
  - weather-forecasting
  - foundation-model
  - extreme-weather
  - frequency-domain
  - memory-network
  - swin-transformer
  - arxiv-2025
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# UniExtreme

**UniExtreme** 是首个面向多样化极端天气预测的通用基础模型，由 Ni, Zhang & Liu（HKUST 广州）提出（arXiv:2508.01426, 2025）[^src-uniextreme]。它同时利用 18 种真实极端天气事件标注数据和一般气象数据进行训练，无需额外微调即可预测各类极端事件。

## 核心问题

现有天气基础模型（[[aurora|Aurora]] 领域不同；GraphCast、PanguWeather 等为一般天气优化）在极端天气预测上存在显著性能差距。此前极端天气方法仅处理单一事件类型（如 NowcastNet 仅处理极端降水，FuXiExtreme 仅处理极端降雨和风力），无法泛化[^src-uniextreme]。

## 架构

```
Input Weather Grid
    ↓
Region Partitioning (uniform spatial partition)
    ↓
Event Prior Augmentation (EPA)  ← extreme event memory pool
    ↓
Adaptive Frequency Modulation (AFM)  ← Beta filters + band aggregation
    ↓
Region Merging
    ↓
Swin Transformer Backbone (FuXi-based)
    ↓
Predicted Weather Grid (next hour)
```

### 自适应频率调制（AFM）

[[adaptive-frequency-modulation|AFM]] 模块通过可学习 Beta 分布滤波器和多粒度频段聚合，捕获正常与极端天气区域间的频谱差异[^src-uniextreme]：

- **Beta 滤波器**：N=10 个区域自适应滤波器，模式（mode）通过频段对数划分手动指定，展布（spread）通过线性变换学习
- **频段聚合**：CNN + 时间嵌入 → 时空感知权重 → softmax 加权聚合多频段信号
- **对数频段策略**：低频段更细粒度（增长率 γ=1.3），匹配天气数据低频主导的特性

### 事件先验增强（EPA）

[[event-prior-augmentation|EPA]] 模块通过分类记忆池和双层融合网络处理极端事件的层次化和复合性[^src-uniextreme]：

- **记忆构建**：从 2022 年真实极端事件中提取区域天气状态，KMeans 聚类标准化为固定容量 U=5
- **类内融合**：每个区域独立查询各类型记忆，通过注意力机制聚合
- **类间融合**：类内融合结果再通过注意力机制整合为混合类型记忆
- **残差增强**：混合记忆通过残差连接增强原始输入

## 数据集

**HR-Extreme-V2**（本文构建）：26TB，覆盖美国本土 2019-2024，6km 分辨率（530×900 网格），69 个大气变量，18 种极端事件类型（包括 Flood、Tornado、Hail、Flash Flood、Heat、Cold、Lightning 等）[^src-uniextreme]。

## 性能

| 指标 | UniExtreme | 最佳基线 | 改进 |
|------|-----------|----------|------|
| Ext. MAE (×e-2) | 8.04 | 9.28 (OneForecast) | ~11% |
| Ext. RMSE (×e-2) | 12.62 | 14.31 (FuXi) | ~10% |
| Gap MAE (×e-2) | 2.88 | 3.06 (OneForecast) | ~6% |

MSL 变量上正常-极端差距缩小约 37%（vs 最佳基线）[^src-uniextreme]。

## 与相关模型的对比

| 维度 | UniExtreme | GraphCast/Pangu | [[most|MoST]] | [[aurora|Aurora]] |
|------|-----------|-----------------|---------------|-------------------|
| 领域 | 极端天气 | 一般天气 | 交通 ST | 通用 TS |
| 极端监督 | 18 种真实事件 | 无 | 无 | 无 |
| 频域处理 | Beta 滤波 + 聚合 | 无 | 无 | 无 |
| 记忆/先验 | EPA 记忆池 | 无 | 无 | 原型引导 |
| 生成方式 | 判别式 | 判别式 | 判别式 | 生成式 (Flow Matching) |
| 多模态 | 否 | 否 | 是 | 是 |

## 相关页面

- [[source-uniextreme]] — 源文件摘要
- [[adaptive-frequency-modulation]] — AFM 技术
- [[event-prior-augmentation]] — EPA 技术
- [[extreme-weather-forecasting]] — 极端天气预测概念
- [[most]] — MoST 多模态 ST 基础模型（不同领域）
- [[aurora]] — Aurora 生成式多模态 TS 基础模型（不同领域）
- [[simdiff]] — SimDiff 扩散模型（频域相关）
- [[traffic-forecasting]] — 交通预测（事故感知相关）

[^src-uniextreme]: [[source-uniextreme]]
