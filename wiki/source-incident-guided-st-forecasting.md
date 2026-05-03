---
title: "Incident-Guided Spatiotemporal Traffic Forecasting"
type: source-summary
tags:
  - traffic-forecasting
  - incident-aware
  - spatial-temporal
  - kdd-2026
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# Incident-Guided Spatiotemporal Traffic Forecasting

Fan, Li, Zou, Ye & Du (Beihang University). KDD 2026. arXiv:2602.02528v2.

## 核心论点

现有交通预测模型仅从历史数据中捕获时空依赖，忽视了突发交通事件（事故、恶劣天气等）作为外部扰动对时空模式的显著改变。非重复性事件的影响不可预测，使其难以从历史序列中观察模式。IGSTGNN 通过显式建模事件影响的时空传播与衰减来解决此问题[^src-incident-guided-st-forecasting]。

## 方法

提出 **IGSTGNN**（Incident-Guided Spatiotemporal Graph Neural Network），基于传统时空建模骨干网络，引入两个核心模块：

1. **[[incident-context-spatial-fusion|ICSF]]** — 通过注意力机制融合事件特征、传感器属性和空间关系张量 D，为每个网络节点生成事件感知的初始状态表示，捕获非均匀空间影响[^src-incident-guided-st-forecasting]。
2. **[[temporal-incident-impact-decay|TIID]]** — 使用高斯衰减函数显式建模事件影响随时间的动态消散过程，将衰减后的事件影响叠加到基础预测趋势上[^src-incident-guided-st-forecasting]。

时空建模模块采用解耦设计，分别捕获外部传播影响和节点固有趋势，并使用多图卷积（静态 + 自适应 + 动态邻接矩阵）[^src-incident-guided-st-forecasting]。

## 数据集

基于 XTraffic 基准构建三个子数据集（California 2023），5 分钟粒度，包含事件记录与交通时间序列的时间对齐标注：

| 数据集 | 节点 | 边 | 事件数 |
|--------|------|----|--------|
| Alameda | 521 | 13,828 | 14,687 |
| Contra Costa | 496 | 13,339 | 5,587 |
| Orange | 990 | 29,142 | 18,700 |

任务：过去 1 小时（12 步）→ 预测未来 1 小时（12 步）[^src-incident-guided-st-forecasting]。

## 实验结果

- IGSTGNN 在三个数据集上全面超越 SOTA，Alameda 平均 MAE 比次优低 5.65%[^src-incident-guided-st-forecasting]。
- 长期预测（Horizon 12）优势显著，Orange 上仍保持 4.1% MAE 优势，验证 TIID 的有效性[^src-incident-guided-st-forecasting]。
- ICSF 和 TIID 作为即插即用模块集成到 AGCRN、GWNET、STTN、D2STGNN 后均带来一致提升，双模块组合时协同效应最强（D2STGNN 上 MAE 提升 13.23%）[^src-incident-guided-st-forecasting]。

## 局限性

- 仅处理最新时间步发生的事件，历史事件的持续影响假设已隐含在近期交通数据中[^src-incident-guided-st-forecasting]。
- 空间关系张量 D 是预定义的，未学习动态空间关系[^src-incident-guided-st-forecasting]。
- 未来工作方向：扩展到更广泛的外部扰动，利用基础模型或 agentic pipeline 提取和规范化事件语义[^src-incident-guided-st-forecasting]。

## 与相关工作的关系

- 与 [[conformer|ConFormer]]（KDD 2026）同属事件感知交通预测，但 ConFormer 聚焦事故场景的事件图传播 + GLN，IGSTGNN 聚焦更广泛的非重复性事件 + 注意力空间融合 + 时间衰减[^src-incident-guided-st-forecasting]。
- 相比 DIGC-Net 的隐式特征融合，IGSTGNN 显式建模时空传播与衰减过程[^src-incident-guided-st-forecasting]。

[^src-incident-guided-st-forecasting]: [[source-incident-guided-st-forecasting]]
