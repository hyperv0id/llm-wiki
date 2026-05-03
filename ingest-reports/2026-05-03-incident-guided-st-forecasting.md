# Ingest 报告：IGSTGNN — Incident-Guided Spatiotemporal Traffic Forecasting

## 创建
- wiki/source-incident-guided-st-forecasting.md — WHY：KDD 2026 论文源文件摘要，提出 IGSTGNN 框架显式建模事件影响的时空传播与衰减
- wiki/igstgnn.md — WHY：新实体页面，记录 IGSTGNN 模型的架构、关键结果和与 ConFormer 的对比
- wiki/incident-context-spatial-fusion.md — WHY：新技术页面，详解 ICSF 模块的注意力融合 + 空间关系张量机制
- wiki/temporal-incident-impact-decay.md — WHY：新技术页面，详解 TIID 模块的高斯衰减建模机制

## 修改
- wiki/accident-aware-traffic-forecasting.md — WHY：扩展概念范围从 accident-aware 到更广的 incident-aware，添加 IGSTGNN 方法段落、挑战点、数据集和对比表；更新 confidence 从 medium→high（新增第二个源文件支撑）；更新 source_count 1→2
- wiki/traffic-forecasting.md — WHY：添加 Incident-Guided 方法段落和 IGSTGNN 引用，添加 XTraffic 数据集引用，更新 source_count 13→14
- wiki/large-scale-spatial-temporal-graph.md — WHY：添加 XTraffic 数据集信息和引用，更新 source_count 2→3
- wiki/conformer.md — WHY：添加交叉链接到 IGSTGNN（同为 KDD 2026 事件感知交通预测模型）

## 新建交叉链接
- [[igstgnn]] ↔ [[conformer]]
- [[igstgnn]] ↔ [[accident-aware-traffic-forecasting]]
- [[igstgnn]] ↔ [[traffic-forecasting]]
- [[incident-context-spatial-fusion]] ↔ [[guided-layer-normalization]]
- [[incident-context-spatial-fusion]] ↔ [[adaptive-graph-agent-attention]]
- [[temporal-incident-impact-decay]] ↔ [[accident-aware-traffic-forecasting]]
- [[large-scale-spatial-temporal-graph]] ↔ [[source-incident-guided-st-forecasting]]
