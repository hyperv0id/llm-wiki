---
title: "LLMGeovec: Geolocation Representation from LLMs are Generic Enhancers for Spatio-Temporal Learning"
type: source-summary
tags:
  - llm
  - geolocation
  - spatial-temporal
  - representation-learning
  - geographic-prediction
  - time-series-forecasting
  - aaai-2025
created: 2026-06-18
last_updated: 2026-06-18
source_count: 1
confidence: medium
status: active
---

# LLMGeovec 论文摘要

**LLMGeovec**（Geolocation Representation from Large Language Models are Generic Enhancers for Spatio-Temporal Learning）由香港理工大学 Junlin He、Tong Nie、Wei Ma 发表于 AAAI 2025。论文提出了一种无需训练的方法，利用 LLM 和 OpenStreetMap 辅助地图数据生成通用地理定位表示（LLMGeovec），作为时空学习的增强器[^src-geolocation-llm-st]。

## 核心贡献

1. **训练无关的通用地理定位表示**：通过 LLM 和 OpenStreetMap 数据直接生成坐标的语义嵌入，无需任何微调或训练，实现了全球覆盖。

2. **即插即用的时空学习增强器**：LLMGeovec 通过简单的特征拼接即可集成到各种时空学习模型中，在地理预测（GP）、长期时间序列预测（LTSF）和基于图的时空预测（GSTF）三个任务上均实现性能提升。

3. **LLM 地理知识提取**：验证了 LLM 内部蕴含丰富的地理空间知识，通过平均最后一层隐藏状态即可提取高质量的地理表示。

## 核心方法

### 提示生成

对于给定坐标，通过 OpenStreetMap 生成结构化提示，包括：
- 逆地理编码获得地址层级（国家、省、市、街道）
- Overpass API 获取 100 公里范围内最近的 10 个 POI

### 文本嵌入

使用预训练 LLM（LLaMa3 8B 或 Mistral 8x7B）处理提示，取其最后一层全部 token 的平均嵌入作为 LLMGeovec 表示。无需修改模型或重复提示[^src-geolocation-llm-st]。

### 下游任务集成

- **GP**：直接用 LLMGeovec 或与其他地理表示拼接，通过岭回归预测社会经济和气候指标（贫困率、人口密度、温度等）。
- **LTSF**：将 LLMGeovec 通过两层 MLP 适配器降维后，与时间序列嵌入拼接，增强节点区分和空间关系建模。
- **GSTF**：将 LLMGeovec 作为 GNN 中的节点特征增强，捕捉先验空间语义。

## 实验结果

### 地理预测（GP）

构建了跨 3 个尺度（全球/国家/城市）、14 个任务的多主题基准。LLMGeovec（LLaMa3 8B）表现：
- 全球尺度：14 项任务中 13 项 R² > 0.75，多数 > 0.90
- 国家尺度：超越依赖街景和 GNN 端到端训练的 SOTA 模型（MapillaryGCN），R² 提升 0.07–0.10
- 城市尺度：在 NYC 的贫困率、教育水平、收入水平、犯罪率上超越或持平基于人类活动数据和复杂图学习的方法

### 长期时间序列预测（LTSF）

在 6 个数据集上，LLMGeovec 使 iTransformer、TSMixer、RMLP、Informer 等模型的 MSE 平均降低 2.47%–22.01%，其中 Traffic-SD 数据集改善最显著（22.01%）[^src-geolocation-llm-st]。

### 图时空预测（GSTF）

在 LargeST 基准上，DCRNN、STGCN、ASTGCN、AGCRN、GWNET、MTGNN 等主流 STGNN 均受益于 LLMGeovec。更值得注意的是，简单的 MLP + LLMGeovec 即可达到与 GNN 方法相当的性能，最高提升 26.53%。

### 零样本迁移

在 LaDe 数据集上，LLMGeovec 在零样本跨区域迁移场景中表现优于可学习嵌入（STID），展示了其内在通用地理知识的泛化能力。

## 局限性

- 目前仅测试了 LLaMa3 8B 和 Mistral 8x7B，更大 LLM 的效果有待验证
- 提示设计对性能有显著影响（地址信息最为关键，POI 数量需适中）
- 计算开销虽小但需额外调用 LLM 进行嵌入生成

[^src-geolocation-llm-st]: [[source-geolocation-llm-st]]