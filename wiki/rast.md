---
title: "RAST"
type: entity
tags:
  - traffic-forecasting
  - retrieval-augmented
  - spatio-temporal
  - rag
  - aaai-2026
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# RAST

RAST（Retrieval-Augmented Spatio-Temporal forecasting）是首个将检索增强生成（RAG）范式引入时空预测的通用框架，由 HKUST-GZ /CUHK 团队提出，发表于 AAAI 2026。[^src-rast]

## 动机

现有 STGNN 面临两个核心瓶颈：[^src-rast]
1. **有限上下文容量**：预训练 STGNN 的嵌入容量受限于模型参数规模，无法充分编码大规模路网中的复杂时空依赖。
2. **低可预测性点**：由于时空异质性，细粒度时空点（如特定时段特定路段的交通模式）难以通过复杂架构捕获，而增加模型复杂度又导致计算成本膨胀。

灵感来源于 NLP 中 RAG 解决 LLM 长尾知识问题的成功经验，RAST 提出：与其让模型参数记住所有复杂模式，不如通过外部记忆显式存储和检索。[^src-rast]

## 架构

RAST 由五个核心组件构成（详见 [[source-rast|source-summary]]）：[^src-rast]

| 组件 | 功能 |
|------|------|
| 解耦编码器 | 2D 卷积时间编码 + 图变换空间编码，产出双维度嵌入 |
| 查询生成器 | 残差 FFN 融合时空嵌入，构造上下文感知检索查询 |
| 时空检索存储 | FAISS 索引的双维度记忆库，存储向量化历史模式 |
| ST-Retriever | L2 距离 Top-k 检索 + 信息熵动量评分 |
| 主干预测器 | 交叉注意力融合 + MLP/预训练 STGNN 预测 |

## 关键实验

在 6 个数据集（PEMS03/04/07/08 + LargeST SD/GBA）上对比 21 个基线，RAST 取得 SOTA。[^src-rast]
- PEMS07：MAE 19.52，超越 DSTAGNN 8.87%
- PEMS08：RMSE 23.33，超越 STKD 1.65
- SD 数据集：MAE 18.39，RMSE 31.96，MAPE 12.19%
- GBA 大规模（2,352 节点）：全维度超越 DSTAGNN 和 RPMixer

## 效率

RAST 在大规模数据集上训练速度最快（GBA 154.08 秒/epoch），推理速度最快（43.52 秒），内存仅 3.71 GB。[^src-rast] 对比：D2STGNN 需 45.10 GB / 5392.56 秒/epoch。RAST 将计算成本维持在 STGCN 级别，同时性能显著优于复杂模型。[^src-rast]

## 消融研究

| 移除组件 | MAE 退化 | 关键发现 |
|----------|----------|----------|
| 查询生成器 | 25.6% | 最关键的组件 |
| 时间编码器 | 21.2% | 双流编码不可或缺 |
| 空间编码器 | 17.2% | 双流编码不可或缺 |
| ST-Retriever | 11.2% | 检索机制贡献显著 |
| MLP 预测器 | 11.0% | 预测器设计重要 |

仅使用检索嵌入而不融合查询时，MAE 反而提升（19.38 vs 19.52），但 MAPE 退化 61.8%，说明检索能捕获整体量级但丢失分布细节。[^src-rast]

## 理论

RAST 的信息论基础：传统 STGNN 参数 $\theta$ 能捕获的信息受限于 $I(X;Y|\theta) \leq H(\theta)$，引入外部记忆 $M$ 后将容量扩展为 $I(X;Y|\theta,M) \leq H(\theta) + H(M)$。[^src-rast] 检索复杂度 $O(k \log M + kd)$，远低于图注意力 $O(N^2)$。

## 相关页面

- [[retrieval-augmented-spatio-temporal-forecasting]] — RAG-for-STF 范式
- [[spatio-temporal-retrieval-store]] — 双维度记忆库 + FAISS
- [[dual-dimension-feature-disentanglement]] — 时空特征解耦
- [[gtr]] — 全局时间检索，同为检索增强的时序预测方法
- [[ragc]] — 大规模路网正则化自适应图卷积
- [[traffic-forecasting]] — 交通预测方法概览

[^src-rast]: [[source-rast]]