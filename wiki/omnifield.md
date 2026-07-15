---
title: "OmniField"
type: entity
tags:
  - multimodal-spatiotemporal
  - neural-field
  - conditioned-neural-field
  - cross-modal-fusion
created: 2026-07-21
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

# OmniField

**OmniField**（Valencia et al., ICLR 2026）是一个连续性感知的多模态条件化神经场框架，用于从稀疏、不规则、噪声污染的科学观测数据中学习连续时空场[^src-omnifield]。由 UCLA、Columbia University 和 Brookhaven National Laboratory 联合提出。

## 核心能力

OmniField 统一处理四类任务，无需 gridding 或代理预处理[^src-omnifield]：

1. **重建（Reconstruction）**：从观测值预测同一位置同一时刻的值
2. **空间插值（Spatial Interpolation）**：预测未观测位置的值
3. **预测（Forecasting）**：预测未来时刻的值
4. **跨模态预测（Cross-modal Prediction）**：预测输入中未出现的模态

## 架构

采用编码器-处理器-解码器架构[^src-omnifield]：

- **编码器**：基于 Gaussian Fourier Features + 正弦初始化的 query-local 置换不变编码
- **处理器**：[[multimodal-crosstalk|多模态串扰（MCT）]] + [[iterative-cross-modal-refinement|迭代跨模态精炼（ICMR）]] 对齐跨模态信号
- **解码器**：每模态独立轻量解码器
- **Fleximodal Fusion**：通过模态存在掩码适应任意输入子集

## 关键结果

- 跨基准平均相对误差降低 22.4%，超越 8 个基线（UNet, ResNet, FNO, OFormer, CORAL, PROSE-FD, MIA, SCENT）[^src-omnifield]
- 严重传感器噪声下性能接近干净输入水平[^src-omnifield]
- EPA-AQS 真实数据上模态越多性能越好（2→4→6）[^src-omnifield]

## 相关模型

- [[source-omnifield|OmniField 论文摘要]]
- [[multimodal-crosstalk]]
- [[iterative-cross-modal-refinement]]
- [[fleximodal-fusion]]

[^src-omnifield]: [[source-omnifield]]
