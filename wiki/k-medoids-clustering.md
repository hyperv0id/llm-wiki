---
title: "K-medoids Clustering"
type: technique
tags:
  - clustering
  - retrieval
  - memory-bank
  - time-series-forecasting
created: 2026-08-19
last_updated: 2026-08-19
source_count: 1
confidence: low
status: active
---

# K-medoids Clustering

**K-medoids** 是一种划分式聚类算法，与 K-means 类似但要求每个簇的中心必须是数据集中的真实样本（medoid），而不是由样本均值生成的合成中心。该特性使其在需要保留原始数据语义或序列结构的场景中优于 K-means——例如，将历史时间序列样本压缩为固定大小的记忆库时，medoid 本身就是一段真实、连贯的过去序列[^src-predicting-the-future-by-retrieving-the-past-aaai2026]。

## 与 K-means 的区别

| 特性 | K-means | K-medoids |
|------|---------|-----------|
| 簇中心 | 样本均值（可能不在数据集中） | 真实样本 medoid |
| 目标函数 | 到均值的平方欧氏距离之和 | 到 medoid 的（可定制）距离之和 |
| 典型距离 | 欧氏距离 | 欧氏、曼哈顿或其他距离 |
| 可解释性 | 中心为合成点 | 中心为真实观测 |

在时间序列预测的记忆库构建中，K-means 产生的均值中心可能对应一条不真实的历史轨迹，而 K-medoids 保证记忆库中的每个代表序列都是实际出现过的模式[^src-predicting-the-future-by-retrieving-the-past-aaai2026]。

## 在 PFRP 中的应用

[[pfrp|PFRP]] 用 K-medoids 对训练样本做聚类，仅保留 K 个 medoid 样本作为 [[global-memory-bank|Global Memory Bank]] 的 key-value 条目。论文报告在 Electricity 数据集上 K-medoids 步骤耗时约 52 秒（PCL 训练 134 秒），GMB 构建总耗时 186 秒[^src-predicting-the-future-by-retrieving-the-past-aaai2026]。

## 相关

- [[global-memory-bank]] — PFRP 中基于 K-medoids 构建的固定大小历史模式库
- [[pfrp]] — 使用 K-medoids 构建 GMB 的检索增强预测框架
- [[predictive-contrastive-learning]] — PFRP 中用于学习检索特征表示的对比学习策略

[^src-predicting-the-future-by-retrieving-the-past-aaai2026]: [[source-predicting-the-future-by-retrieving-the-past-aaai2026]]
