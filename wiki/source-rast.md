---
title: "RAST — Retrieval-Augmented Spatio-Temporal Framework for Traffic Prediction"
type: source-summary
tags:
  - traffic-forecasting
  - retrieval-augmented
  - spatio-temporal
  - rag
  - faiss
  - aaai-2026
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# RAST — Retrieval-Augmented Spatio-Temporal Framework for Traffic Prediction

Ruan, Dang, Zhou, Lyu & Liang (HKUST-GZ / CUHK), AAAI 2026. arXiv:2508.16623. [^src-rast]

## 核心贡献

RAST 是首个将 RAG（Retrieval-Augmented Generation）范式引入时空预测的通用框架。[^src-rast] 论文针对当前 STGNN 的两大瓶颈：(i) 有限上下文容量难以建模复杂时空依赖；(ii) 时空异质性导致细粒度点的低可预测性。[^src-rast]

## 架构设计

RAST 由五个核心组件构成：[^src-rast]

1. **解耦编码器（Decoupled Encoder）**：分别用 2D 卷积处理时间维度、图变换处理空间维度，产出解耦的 $E_{tp}$ 和 $E_{sp}$ 嵌入。
2. **上下文感知查询生成器（Query Generator）**：将时空嵌入拼接投影后经 L 层残差 FFN 生成融合查询 $Q_{st}$。
3. **时空检索存储（ST-Retrieval Store）**：基于 FAISS 维护双维度记忆库 $M = \{M_{sp}, M_{tp}\}$，存储向量化历史模式及元数据。
4. **ST-Retriever**：通过 L2 距离执行 Top-k 检索，利用信息熵计算动量分数，实现加权模式聚合。
5. **通用主干预测器（Universal Backbone Predictor）**：兼容预训练 STGNN 或简单 MLP，通过交叉注意力融合检索嵌入与查询。

## 关键结果

在 6 个数据集（PEMS03/04/07/08 + LargeST SD/GBA）上对比 21 个基线模型，RAST 取得 SOTA。[^src-rast] 消融实验显示查询生成器最为关键（移除后 MAE 退化 25.6%），时空编码器次之（空间 17.2%、时间 21.2%）。[^src-rast] 效率方面，RAST 在大规模数据集（GBA 2,352 节点）上训练速度 154.08 秒/epoch，推理 43.52 秒，且内存仅 3.71 GB，显著优于 D2STGNN（45.10 GB / 5392.56 秒/epoch）。[^src-rast]

## 理论支撑

RAST 的信息论基础：传统 STGNN 的互信息受限于 $I(X;Y|\theta) \leq H(\theta)$，RAST 通过外部记忆 $M$ 将容量扩展为 $I(X;Y|\theta,M) \leq H(\theta) + H(M)$。[^src-rast] 检索复杂度 $O(k \log M + kd)$，相比图注意力 $O(N^2)$ 显著降低。[^src-rast]

## 局限性

1. 冷启动场景下初始记忆库构建依赖历史数据质量。[^src-rast]
2. 对完全新颖场景（记忆库中无相似历史模式）适应性有限。[^src-rast]
3. 记忆更新阶段引入额外开销，可能限制资源受限环境下的实时部署。[^src-rast]
4. 需要领域特定的超参数调优（相似度阈值、更新间隔）。[^src-rast]

[^src-rast]: [[source-rast]]