---
title: "Source: PDFormer (AAAI 2023)"
type: source-summary
tags:
  - traffic-forecasting
  - transformer
  - source-summary
created: 2026-08-30
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# PDFormer: Propagation Delay-Aware Dynamic Long-Range Transformer for Traffic Flow Prediction

对应 `raw/pdformer-jiang-2023.pdf`（arXiv 2301.07945v3，水印显示 v3 更新于 2024-03-07；正文为 AAAI 2023 会议版，AAAI-23 卷 37 号 4）[^src-pdformer-jiang-2023]。

## 基本信息

- 作者：Jiawei Jiang、Chengkai Han（共同一作）、Wayne Xin Zhao、Jingyuan Wang（通讯作者，jywang@buaa.edu.cn）
- 机构：北京航空航天大学计算机学院；鹏城实验室；北航经济管理学院；中国人民大学高瓴人工智能学院
- 代码：github.com/BUAABIGSCity/PDFormer

## 核心论点

论文主张 GNN 系交通预测模型存在静态空间依赖、缺失长程依赖（over-smoothing）、忽略传播时延三条局限，并提出以 self-attention 为核心的 PDFormer 逐一对应：地理/语义双掩码的空间自注意力同时覆盖短程与长程依赖，延迟感知特征变换（k-Shape 模式库加权更新 key）显式建模传播时延，时间自注意力捕获逐节点动态时间模式[^src-pdformer-jiang-2023]。

## 主要结果

论文报告在 6 个数据集（PeMS04/07/08 多步；NYCTaxi、CHIBike、T-Drive 单步）上以 MAE/MAPE/RMSE 全面优于 17 个基线（t-test 0.01），相对第二名平均提升 4.58%/5.00%/4.79%；PeMS04 每 epoch 训练/推理 133.871s/8.120s，比 ASTGNN 快 35%/80% 以上。消融验证 SSA 优于 GCN、双掩码与 DFT 均为必要；注意力可视化显示掩码使模型聚焦近邻与远处功能相似区域[^src-pdformer-jiang-2023]。

## 局限

论文未报告参数量、显存或与轻量模型的效率对比（这些数字由后续论文补充测量，见 [[pdformer]]）；DFT 与语义掩码依赖历史数据与多个超参（λ、K、S、聚类数）。论文自述未来方向为风电预测等其他时空任务与预训练[^src-pdformer-jiang-2023]。

[^src-pdformer-jiang-2023]: [[source-pdformer-jiang-2023]]
