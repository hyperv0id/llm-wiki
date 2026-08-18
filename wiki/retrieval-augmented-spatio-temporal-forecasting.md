---
title: "Retrieval-Augmented Spatio-Temporal Forecasting"
type: concept
tags:
  - retrieval-augmented
  - spatio-temporal
  - rag
  - traffic-forecasting
created: 2026-06-08
PUT last_updated: 2026-08-19
PUT source_count: 6
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
- **[[trace|TRACE]]**（NeurIPS 2025）是首个**跨模态**时序检索器——此前方法（RAST/RATD/GTR/PIR）均仅用单模态时序嵌入做检索。TRACE 将时序嵌入与对齐文本上下文锚定到共享语义空间，支持 Text→TS / TS→Text / TS→TS 检索，并通过 soft prompt 增强冻结 TSFM（Time-MoE / Timer-XL / Moment）[^src-trace-neurips2025]。
- **[[pir|PIR]]**（Post-forecasting Identification and Revision，Liu et al., NeurIPS 2025）是纯时序的轻量实例检索修订：后处理阶段在训练输入–目标对上以实例归一化编码 + 余弦相似度检索 top-K 相似实例、softmax 加权求和作为全局修订项，不经生成模型；与 RAST 的双维 FAISS 检索、RATD 的扩散参照引导均不同[^src-pir]。

## 开放问题

1. 检索增强范式是否可推广到其他时空任务（如插补、异常检测）？[^src-rast]
2. 多模态检索增强（天气、事件、文本）是否进一步提升预测精度？[^src-rast]
3. 联邦检索增强——跨组织共享检索库同时保护数据隐私。[^src-rast]
- **[[ts-memory|TS-Memory]]**（KDD 2026）提出检索到参数蒸馏范式——将在线 kNN 检索的预测分布知识离线蒸馏为轻量参数模块，推理时无需检索、$O(1)$ 复杂度，解决在线检索的推理延迟问题[^src-ts-memory]。

[^src-rast]: [[source-rast]]
[^src-ratd]: [[source-ratd]]
- **[[pfrp|PFRP]]**（AAAI 2026）是检索增强的单变量时序预测框架：通过 PCL 训练编码器 + K-medoids 聚类构建固定大小 Global Memory Bank，推理时按特征余弦相似度检索 top-k 历史模式，经 confidence gate 和 output gate 调制后与局部预测动态融合。与 RAST 的双维度时空检索不同，PFRP 仅在时间维度检索；与 RATD 的扩散参照引导不同，PFRP 不依赖扩散模型[^src-predicting-the-future-by-retrieving-the-past-aaai2026]。
[^src-pir]: [[source-pir]]
[^src-trace-neurips2025]: [[source-trace-neurips2025]]
[^src-predicting-the-future-by-retrieving-the-past-aaai2026]: [[source-predicting-the-future-by-retrieving-the-past-aaai2026]]
## 平稳性感知的检索增强

- **[[saraf|SARAF]]**（KDD 2026）从非平稳性角度改进检索增强预测：通过诊断实验证明相似度检索的可靠性随平稳性变化（Spearman ρ 从 1.000 降至 0.285），提出时间对齐增强 + 平稳性控制的多样性 MMR + 自适应 Gaussian 聚合。与 RAST 的双维 FAISS 检索不同，SARAF 在纯时间序列上通过平稳性分数 s̄ 自适应调节检索策略。[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]
[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]: [[source-stationarity-aware-retrieval-augmented-forecasting-kdd26]]
[^src-ts-memory]: [[source-ts-memory-time-series-foundation-models-kdd26]]
