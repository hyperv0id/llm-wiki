---
title: "Source: WorldTS — World Modeling for Multimodal Covariate-aware Time Series Forecasting"
type: source-summary
tags:
  - time-series
  - forecasting
  - multimodal
  - covariate
  - world-model
created: 2026-10-13
last_updated: 2026-10-13
source_count: 1
confidence: medium
status: active
---

# WorldTS: World Modeling for Multimodal Covariate-aware Time Series Forecasting

**来源**：Zhu, Qiu, Cheng, Shen, Guo, Yang, Hu（华东师范大学）与 C. S. Jensen（Aalborg University），arXiv:2609.31162v1 [cs.LG]，2026-09-25，preprint。PDF 存于 raw/。

**核心论点**：观测只是系统动力学的部分体现，观测空间直接预测把状态演化与观测生成纠缠在一起；已有潜空间预测（LatentTSF、TimeAlign）仍只从目标序列自身编码，外部多模态信息未参与状态预测。WorldTS 把数值/文本/图像协变量编码为潜表征，直接条件化潜空间未来状态预测，并以编码真值未来观测所得状态直接监督预测状态。

**贡献**：(1) 把协变量感知预测重构为潜空间未来状态预测问题；(2) 两阶段训练——Stage 1 学协变量条件化的潜动力学（L1 + cosine state matching、stop-gradient、VICReg 防塌缩正则；文本用冻结 GPT-2 embedding、图像用轻量 CNN、融合用 CFA 低秩残差），Stage 2 冻结编码器与 predictor、只训解码器；(3) 21 个真实数据集上的实验：12 个数值协变量数据集平均 MSE/MAE 全部最低，8 个 Time-MMD 文本域 16 组对比中 14 组第一，MMSP 图像协变量预测误差最低。

**局限**（附录 C）：协变量条件化状态学习缺乏理论刻画；目标与正则沿用通用设计，论文将其定位为世界模型思想向时序预测移植的初步尝试；仅规则采样、逐数据集训练、未接预训练基础模型。Table 5 逐 horizon 少数设定非第一（FR@24/DE@24 被 KITE、Sdwpfm1/2@24 被 DAG 超过）。

## Related Pages

- [[worldts]] — 模型页
- [[latent-space-future-state-prediction]] — 核心技术页

[^src-worldts]: [[source-worldts]]
