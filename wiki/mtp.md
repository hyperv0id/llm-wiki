---
title: "MTP"
type: entity
tags:
  - multimodal
  - time-series
  - traffic
  - classification
  - frequency-domain
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# MTP (Multimodal Traffic Profiling)

MTP 是一个面向城市交通状态分类的多模态框架，由 Xiang et al. (Nanjing University of Information Science and Technology / Nanjing University / Macquarie University / University of Auckland) 提出，发表于 AAAI 2026[^src-mtp]。

## 核心思路

传统交通信号分析方法主要依赖单模态数值数据，忽略了多模态异构数据中的语义信息。MTP 通过三大模态编码器在**频域**中进行学习：

1. **数值**：通过频域 MLP 处理原始时间序列数据
2. **视觉**：将时间序列增强为频率图像和周期性图像，通过多尺度卷积 + FIR 频谱压缩提取特征
3. **文本**：利用 LLM 生成描述性文本（主题、背景、项目描述），编码后转为频域表示

三模态通过**分层对比学习**（监督 + InfoNCE 无监督 + JS 散度分布对齐）进行融合。

## 关键设计

| 组件 | 功能 |
|------|------|
| 语义嵌入 | 可学习权重向量 ψ 将原始输入映射到高维隐藏表示 |
| 频域 MLP | 复数权重矩阵 W = W_i + ηW_j，在频域非线性变换 |
| 视觉增强 | FFT → 周期编码 → 多尺度 1D+2D 卷积 → 双线性插值 → 图像 |
| FIR 频谱压缩 | Hamming 窗 FIR 滤波器组 + 平均池化，保留核心频率、削弱噪声 |
| 跨模态增强 | 文本频谱 ⊙ 池化增强图像频谱（反之亦然），双向信息交互 |
| 分层对比融合 | α·监督对比 + β·InfoNCE + γ·JS 散度分布对齐 → 加权融合特征 |

## 实验结果

在 6 个交通数据集上全面超越 8 个 SOTA 基线。消融实验显示：视觉分支对波动数据集（DodgerLoop F1 0.585→0.105）贡献最大；三模态各有独立贡献；t-SNE 可视化证实融合特征比单模态更具判别性。

## 与其他模型的关系

- [[multimodal-time-series-forecasting]] — 多模态时间序列预测（MTP 侧重分类而非预测）
- [[multimodal-traffic-profiling]] — 多模态交通状态分析概念
- [[modality-augmentation]] — 从数值序列生成视觉/文本模态的技术
- [[hierarchical-contrastive-fusion]] — 分层对比融合机制
- [[traffic-forecasting]] — 交通预测领域

## 引用

[^src-mtp]: [[source-mtp]]
