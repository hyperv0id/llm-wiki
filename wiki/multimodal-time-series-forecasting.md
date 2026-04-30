---
title: "Multimodal Time Series Forecasting"
type: concept
tags:
  - time-series
  - multimodal
  - forecasting
  - covariate
created: 2026-04-29
last_updated: 2026-04-29
source_count: 2
confidence: high
status: active
---

# Multimodal Time Series Forecasting (多模态时间序列预测)

## 定义

**多模态时间序列预测**是指同时利用多种数据模态（如数值时间序列、图像、文本）进行未来值预测的任务[^src-unca]。

## 背景与挑战

传统时间序列预测方法主要依赖数值型历史数据。然而，在许多实际应用场景中，外部信息以多种形式存在：

| 模态 | 示例 | 挑战 |
|------|------|------|
| 数值 | 温度、销量、价格 | 标准处理 |
| 分类 | 商品ID、门店类型 | 需要 embedding |
| 图像 | 卫星云图、工业检测 | 维度高、语义异构 |
| 文本 | 新闻、天气报告、社交媒体 | 序列长、语义复杂 |

## 现有方法分类

### 1. 文本增强预测

利用 LLM 的强大时间编码能力：
- **Time-LLM**：将时间序列 reprogramming 为 LLM 输入
- **ChatTime**：结合时间感知提示
- **LLM4TS**：零样本 LLM 预测

局限性：通常处理静态文本，难以利用动态文本信息。

### 2. 图像-时间序列预测

- **FusionSF**：专为卫星场景设计
- **MMSP 数据集**：多模态太阳能预测

### 3. 时间序列基础模型方法

| 方法 | 多模态支持 | 特点 |
|------|-----------|------|
| Moirai | 有限 | 展平为联合序列 |
| Chronos | 无 | 仅支持数值 |
| TimesFM | 无 | 仅支持数值 |
| **UniCA** | 完全支持 | 同质化 + 融合 |

## UniCA 的解决方案

UniCA 通过**协变量同质化**将不同模态转换为统一表示：

1. **模态专用编码器**：CNN（图像）、GIST（文本）
2. **线性投影层**：将特征映射到同质时间序列空间
3. **统一融合框架**：Pre-Fusion + Post-Fusion 处理所有模态[^src-unca]

## 数据集

### MMSP (Multimodal Solar Power)

- 太阳能发电预测
- 输入：历史发电量 + 卫星云图
- 评估指标：MAE（MAPE 不稳定）

### Time-MMD

- 多模态时间序列数据集
- 输入：数值序列 + 文本描述
- 评估指标：MAPE

## 实验结果

UniCA 在多模态场景下的表现：

| 数据集 | 基线模型 | +UniCA | 提升 |
|--------|---------|--------|------|
| MMSP | TimesFM | TimesFM | -6.5% MAE |
| MMSP | Chronos-Bolt | Chronos-Bolt | -5.9% MAE |
| Time-MMD | TimesFM | TimesFM | -5.9% MAPE |
| Time-MMD | Chronos-Bolt | Chronos-Bolt | -13.0% MAPE |

## ChannelMTS：高铁信道预测

**ChannelMTS** (KDD 2026) 是首个将**环境信息**（位置、K因子、RMS延迟）融入高铁信道预测的多模态框架[^src-channelmts]。

### 与 UniCA 的区别

| 维度 | UniCA | ChannelMTS |
|------|-------|------------|
| 目标 | TSFMs 协变量适应 | 高铁信道预测 |
| 输入 | 分类/图像/文本协变量 | 环境快照 (位置+K因子+延迟) |
| 核心方法 | 协变量同质化 | 环境-信道对齐 |
| 分布对齐 | RevIN 风格归一化 | 中位数+IQR 归一化 |
| 融合策略 | Pre-Fusion + Post-Fusion | 自适应动态权重 |

### 关键创新

1. **检索增强统计信道 (RAGC)**：从预缓存的高铁地图中检索相似环境对应的历史统计信道
2. **未来环境信息利用**：利用铁路轨迹预定义特性，使用未来环境信息提升预测
3. **线上部署验证**：真实 5G NR 系统上 A/B 测试 MSE 降低 82%-92%

### 实验结果

| 数据集 | ChannelMTS MSE | 最佳基线 MSE | 提升 |
|--------|---------------|-------------|------|
| HSR I | 0.0722 | 0.0859 (ChatTime) | 16% |
| VSR I | 0.1675 | 0.2569 (ChatTime) | 35% |

---

## 相关概念

- [[heterogeneous-covariates]] — 异构协变量
- [[covariate-homogenization]] — 协变量同质化
- [[unified-covariate-adaptation]] — UniCA 框架
- [[channelmts]] — 高铁多模态信道预测框架
- [[timesnet]] — 时间序列基础模型

---

## 引用

[^src-unca]: [[source-unca]]
[^src-channelmts]: [[source-channelmts]]