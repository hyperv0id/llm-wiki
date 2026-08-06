---
title: "Retrieval-Augmented Spatio-Temporal Forecasting"
type: concept
tags:
  - retrieval-augmented
  - spatio-temporal
  - rag
  - traffic-forecasting
created: 2026-06-08
last_updated: 2026-08-06
source_count: 3
confidence: medium
status: active
---

# Retrieval-Augmented Spatio-Temporal Forecasting

检索增强时空预测（Retrieval-Augmented Spatio-Temporal Forecasting）是将 NLP 中 RAG（Retrieval-Augmented Generation）范式迁移到时空预测领域的全新方法论，由 [[rast|RAST]]（Ruan et al., AAAI 2026）首次提出。[^src-rast]

## 核心思想

传统 STGNN 将所有知识压缩进固定容量的模型参数 $\theta$ 中，其捕获的互信息受限于 $I(X;Y|\theta) \leq H(\theta)$。[^src-rast] 检索增强范式通过引入外部记忆 $M$，将信息容量扩展为 $I(X;Y|\theta,M) \leq H(\theta) + H(M)$，从而在不增加模型参数的前提下捕获更复杂的时空依赖。[^src-rast]

## 与 NLP RAG 的类比

| 维度 | NLP RAG | STF RAG（RAST） |
|------|---------|------------------|
| 检索对象 | 文本文档块 | 向量化时空历史模式 |
| 索引工具 | 向量数据库 | FAISS 双维度索引 |
| 查询 | 用户问题嵌入 | 上下文感知融合查询 $Q_{st}$ |
| 融合方式 | LLM 上下文拼接 | 交叉注意力解码 |
| 知识库更新 | 静态或增量 | 动量 EMA 动态更新 |

## 双维度检索

STF 场景的核心挑战在于时空纠缠——一个 $N \times T \times d$ 张量的存储和检索复杂度为 $O(N \cdot T \cdot d)$。[^src-rast] RAST 采用特征解耦策略，将嵌入分解为时间表示 $U \in \mathbb{R}^{T \times d}$ 和空间表示 $V \in \mathbb{R}^{N \times d}$，将存储复杂度降为 $O((N+T) \cdot r)$。[^src-rast] 参见 [[dual-dimension-feature-disentanglement]]。

## 与相关方法的区别

- **[[gtr|GTR]]**（ICLR 2026）：同为检索增强的时序预测方法，但 GTR 仅在时间维度检索全局周期模式（单维度），而 RAST 执行双维度（时间+空间）检索。[^src-rast]
- **[[ragc|RAGC]]**（arXiv 2026）：名称中含 "R"，但 RAGC 是正则化自适应图卷积，并非 RAG 范式——RAST 是首个将 RAG 显式应用于 STF 的工作。[^src-rast]
- **[[uniflow|UniFlow]]**（arXiv 2024）：使用时空记忆检索增强（ST-MRA），但 RAST 的检索基于 FAISS 向量索引和 L2 距离，不同于 UniFlow 的余弦相似度记忆检索。
- RAG 范式同样被引入纯时间序列（非时空图）的生成式预测：[[ratd|RATD]]（NeurIPS 2024）是首个检索增强的时间序列扩散模型，从数据库检索 k 个最近邻参照引导扩散去噪，早于 RAST 将 RAG 应用于预测[^src-ratd]。
- **[[pir|PIR]]**（Post-forecasting Identification and Revision，Liu et al., NeurIPS 2025）是纯时序的轻量实例检索修订：后处理阶段在训练输入–目标对上以实例归一化编码 + 余弦相似度检索 top-K 相似实例、softmax 加权求和作为全局修订项，不经生成模型；与 RAST 的双维 FAISS 检索、RATD 的扩散参照引导均不同[^src-pir]。

## 开放问题

1. 检索增强范式是否可推广到其他时空任务（如插补、异常检测）？[^src-rast]
2. 多模态检索增强（天气、事件、文本）是否进一步提升预测精度？[^src-rast]
3. 联邦检索增强——跨组织共享检索库同时保护数据隐私。[^src-rast]

[^src-rast]: [[source-rast]]
[^src-ratd]: [[source-ratd]]
[^src-pir]: [[source-pir]]