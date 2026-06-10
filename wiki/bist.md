---
title: "BiST"
type: entity
tags:
  - spatiotemporal
  - traffic-forecasting
  - mlp
  - lightweight
created: 2026-06-10
last_updated: 2026-06-10
source_count: 1
confidence: high
status: active
---

# BiST — Bi-directional Spatio-Temporal Prediction

BiST（Bi-directional Spatio-Temporal model）是由中科大提出的轻量级时空预测模型（PVLDB 2025）[^src-bist]。其核心创新在于打破了传统时空模型"输入-标签一致性"的隐含假设，通过双向学习范式显式建模输入与标签之间的时空偏差。

## 动机：时空偏差问题

现有时空预测模型仅进行前向建模——从输入数据提取时空相关性来生成预测，隐含假设输入和标签的时空分布一致。但实际数据中存在显著的**时空偏差**（spatiotemporal deviation）[^src-bist]：

- **空间偏差**：两个节点的输入分布相似但标签分布迥异（反之亦然），模型难以区分
- **时间偏差**：数据出现突然的上升或下降（如交通事故导致的车流激增），模式与平稳时期不同

BiST 通过引入标签信息来显式校正这种偏差。

## 架构

BiST 包含两个过程：

**前向时空学习**（纯 MLP）：
- 时间分解：将输入序列通过移动平均分解为稳定模式（X_l）和趋势模式（X_s），分别用 MLP 处理
- 时空嵌入提示：时间嵌入（一天内时刻 + 周几）+ 自适应节点嵌入
- 基预测生成：多层 MLP → 线性解码器

**后向残差校正**：
- 残差解耦模块：将标签表示分解为虚拟聚类学习的上下文特征和节点个性化特征
- 残差扩散：通过可学习的自适应扩散核在多步内平滑残差
- 校正项生成 → 与基预测相加得到最终预测

## 关键特性

| 特性 | 说明 |
|------|------|
| 参数量 | 极轻量（纯 MLP） |
| 时间复杂度 | 近线性 |
| 训练速度 | SOTA 的 ~54 倍 |
| 显存占用 | SOTA 的 ~7% |
| 大规模支持 | 16,972 节点无 OOM |
| 长期预测 | 20 年跨度优秀表现 |

[^src-bist]: [[source-bist]]
