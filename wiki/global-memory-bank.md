---
title: "Global Memory Bank (GMB)"
type: technique
tags:
  - retrieval-augmented
  - time-series-forecasting
  - memory-bank
  - k-medoids
  - aaai-2026
created: 2026-08-19
last_updated: 2026-08-19
source_count: 1
confidence: low
status: active
---

# Global Memory Bank (GMB)

Global Memory Bank 是 [[pfrp|PFRP]]（AAAI 2026）提出的显式历史模式存储与检索机制，灵感来自 NLP 中的 retrieval-augmented generation 和 memory network。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]

## 结构

GMB 存储 K 个样本对 $\{(\epsilon^{(1)}, y^{(1)}), \dots, (\epsilon^{(K)}, y^{(K)})\}$，其中 $\epsilon^{(i)}$ 是回溯窗口特征（检索 key），$y^{(i)}$ 是对应的预测区间序列（检索 value）。GMB 的回溯窗口长度固定为 96，预测区间长度固定为 720，因此对不同预测长度（96/192/336）只需截取存储序列的前对应步数，无需为每个预测长度重建 GMB。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]

## 构建

1. 用 [[predictive-contrastive-learning|PCL]] 训练 MLP 编码器，将所有训练样本的回溯窗口编码为特征。
2. 在特征空间上做 [[k-medoids-clustering|K-medoids]] 聚类，保留 K 个 medoid 样本。

选择 K-medoids 而非 K-means 的关键原因：K-medoids 使用真实历史样本作为聚类中心，而非合成平均值，确保 GMB 中存储的模式是真实连贯的历史序列。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]

## 检索

推理时将当前回溯窗口编码为 query $\epsilon$，与 GMB 中 K 个 key 计算余弦相似度 $w^{(i)} = \epsilon \cdot \epsilon^{(i)}$，取 top-k 最相似的 key-value 对。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]

## 与其他记忆/检索机制的区别

- **[[spatio-temporal-retrieval-store|RAST 检索库]]**：RAST 使用 FAISS 双维度（时间+空间）向量索引 + 动量 EMA 更新；GMB 是固定大小的单维度（时间）记忆库，构建后不更新。
- **[[ratd|RATD]]**：RATD 从整个训练集检索 k-NN 参照引导扩散去噪；GMB 仅从固定 K 个 medoid 中检索，效率更高。
- **RAFT**：RAFT 在推理时遍历整个训练集检索；GMB 仅检索固定大小记忆库。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]

## 效率

GMB 构建在 Electricity 上耗时 186 秒（PCL 134s + K-medoids 52s），是一次性开销。推理时仅检索固定 K 个样本，复杂度 $O(K)$ 而非 $O(N)$（N 为训练样本数）。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]

[^src-predicting-the-future-by-retrieving-the-past-aaai2026]: [[source-predicting-the-future-by-retrieving-the-past-aaai2026]]
