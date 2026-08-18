---
title: "Predictive Contrastive Learning (PCL)"
type: technique
tags:
  - contrastive-learning
  - time-series-forecasting
  - retrieval-augmented
  - feature-encoding
  - aaai-2026
created: 2026-08-19
last_updated: 2026-08-19
source_count: 1
confidence: low
status: active
---

# Predictive Contrastive Learning (PCL)

PCL 是一种用于时序预测检索的对比学习策略，由 [[pfrp|PFRP]]（AAAI 2026）提出。核心创新在于正样本的选择标准：不基于回溯窗口序列的相似度，而基于预测区间序列的相似度。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]

## 动机

传统对比学习（如 [[contrastive-learning|SimCLR]]）在视觉表征中按数据增强选择正样本。在时序检索场景中，直接用回溯窗口的 MSE 或 DTW 相似度作为正样本标准的问题是：回溯窗口相似并不保证未来也相似。PCL 的直觉是——如果两个样本的预测区间序列更相似，那么它们的回溯窗口编码特征应该更接近，这使得检索到的历史样本对当前预测更有帮助。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]

## 方法

在训练批次 $\{(x^{(1)}, y^{(1)}), \dots, (x^{(B)}, y^{(B)})\}$ 中，对第 $i$ 个样本，其正样本索引为：

$$i^+ = \arg\min_{1 \le j \le B, j \ne i} \|y^i - y^j\|_2^2$$

即选择预测区间序列 MSE 最小的样本。然后用 InfoNCE 目标训练 MLP 编码器：

$$\mathcal{L}_{\text{pcl}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\epsilon^{(i)} \cdot \epsilon^{(i^+)} / \tau)}{\sum_{j=1, j \ne i}^{B} \exp(\epsilon^{(i)} \cdot \epsilon^{(j)} / \tau)}$$

其中 $\epsilon^{(i)}$ 是回溯窗口 $x^{(i)}$ 的编码特征，$\tau$ 是温度参数。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]

## 与其他编码器训练策略对比

论文消融对比了三种策略：[^src-predicting-the-future-by-retrieving-the-past-aaai2026]

- **Predictive Learning (PL)**：编码器附加预测头做标准预测任务（RATD 使用）。PCL 更优，因 PL 优化预测而非检索特征。
- **Contrastive Learning (CL)**：正样本按回溯窗口序列相似度选择。PCL 更优，因 CL 仅保证回溯相似而非未来相似。
- **PCL（本文）**：正样本按预测区间序列相似度选择，直接对齐检索目标。

## 实现

批次大小 256，温度 $\tau = 0.05$，学习率 0.001。训练时排除与 anchor 时间重叠超过 48 个时间戳的样本以防数据泄漏。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]

[^src-predicting-the-future-by-retrieving-the-past-aaai2026]: [[source-predicting-the-future-by-retrieving-the-past-aaai2026]]
