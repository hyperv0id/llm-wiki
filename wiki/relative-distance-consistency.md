---
title: "相对距离一致性"
type: concept
tags:
  - self-supervised
  - representation-learning
  - spatiotemporal
  - metric-learning
created: 2026-07-21
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

# 相对距离一致性（Relative Distance Consistency）

**相对距离一致性**是 [[st-ssdl|ST-SSDL]]（NeurIPS 2025）提出的核心设计原则，用于在无标签条件下将物理空间中的偏差模式迁移到潜在空间[^src-st-ssdl]。

## 定义

> 物理空间中距离近（远）的当前-历史样本对，在潜在空间中也应保持近（远）的相对关系，即 $D_1 > D_2 \Rightarrow \tilde{D}_1 > \tilde{D}_2$。

其中 $D = \|Q^c - Q^a\|_1$（物理空间距离，通过 stop-gradient 固定），$\tilde{D} = \|P^c - P^a\|_1$（潜在空间中正原型间的距离）[^src-st-ssdl]。

## 动机

原始物理空间中，偏差的大小直观可见（如偏差值约 40 vs 约 20（论文 Fig.1(b) 示意图，未标注物理单位）），但映射到高维潜在空间后，对应的潜在距离变得不确定。单纯对原始距离回归面临两个困难：绝对距离尺度在潜在空间中无意义，且连续空间中的距离缺乏结构化的比较基准[^src-st-ssdl]。

相对距离一致性回避了绝对尺度问题，只要求**序关系**得以保留——这是一种比直接回归物理距离更松弛、更易优化的约束。

## 实现

在 SSDL 中，相对距离一致性通过**偏差损失**实现：

$$L_{\text{Dev}} = \left| \nabla(\|Q^c - Q^a\|_1) - \|P^c - P^a\|_1 \right|_1$$

物理空间距离 $\nabla(\|Q^c - Q^a\|_1)$（stop-gradient，视为固定参考）与原型代理距离 $\|P^c - P^a\|_1$ 之间的 L1 差异被最小化。原型作为潜在空间的"标尺"，提供了离散化的距离比较框架[^src-st-ssdl]。

## 与相关概念的关系

- 不同于**度量学习**（metric learning）中常见的绝对距离约束（如 Siamese 网络的最小化同类距离），相对距离一致性只约束序关系。
- 不同于标准的**对比学习**（contrastive learning）中的正/负对分类目标，偏差损失直接对距离值建模。
- 可视为**序数回归**（ordinal regression）在潜在表征空间的一种自监督形式。

## 相关页面

- [[ssdl]] — Self-Supervised Deviation Learning
- [[st-ssdl]] — ST-SSDL 框架
- [[spatiotemporal-deviation]] — 时空偏差
- [[contrastive-learning]] — 对比学习

[^src-st-ssdl]: [[source-st-ssdl]]
