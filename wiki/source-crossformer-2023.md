---
title: "Crossformer: Transformer Utilizing Cross-Dimension Dependency for MTS Forecasting"
type: source-summary
tags:
  - time-series
  - transformer
  - cross-dimension
  - multivariate
  - ICLR-2023
created: 2026-05-30
last_updated: 2026-05-30
source_count: 1
confidence: medium
status: active
---

# Source: Crossformer (Zhang & Yan, ICLR 2023)

**作者**：Yunhao Zhang, Junchi Yan（上海交通大学 / 上海 AI Lab）
**会议**：ICLR 2023
**代码**：https://github.com/Thinklab-SJTU/Crossformer

## 核心论点

现有 Transformer 模型在多变量时间序列 (MTS) 预测中主要建模跨时间依赖 (cross-time dependency)，而忽略了跨维度依赖 (cross-dimension dependency)——即不同变量之间的关联。Crossformer 提出三个组件来显式利用跨维度依赖：Dimension-Segment-Wise (DSW) embedding、Two-Stage Attention (TSA) layer 和 Hierarchical Encoder-Decoder (HED) [^src-crossformer-2023]。

## 主要贡献

1. **DSW Embedding**：将每个变量的时间序列分段嵌入为向量，输出 2D 向量阵列（时间轴 × 维度轴），而非传统方法中将同时间步所有变量点嵌入为单一向量。这保留了维度信息，使跨维度依赖可被显式建模 [^src-crossformer-2023]。

2. **TSA Layer**：分两阶段处理 2D 向量阵列——Cross-Time Stage 对每个维度独立做 MSA 捕获跨时间依赖（$O(DL^2)$）；Cross-Dimension Stage 用 Router 机制（c 个可学习路由向量先聚合再分发）将 $O(D^2L)$ 降至 $O(DL)$。总复杂度 $O(DL^2)$ [^src-crossformer-2023]。

3. **HED**：编码器逐层合并相邻段捕获粗粒度依赖；解码器在每层生成预测并求和，利用多尺度信息 [^src-crossformer-2023]。

## 关键实验结果

- 6 个真实数据集（ETTh1, ETTm1, WTH, ECL, ILI, Traffic），58 个设置中 36 个 top-1、51 个 top-2 [^src-crossformer-2023]
- MTGNN（用 GNN 建模跨维度依赖）优于多数 Transformer 基线，验证跨维度依赖的重要性 [^src-crossformer-2023]
- FEDformer/Autoformer 在 ILI（小数据集）上优于 Crossformer，可能因引入序列分解先验 [^src-crossformer-2023]
- 消融实验：DSW > Transformer baseline; TSA 恒定提升; HED 对长预测有利但短预测略降 [^src-crossformer-2023]

## 局限

1. Cross-Dimension Stage 建立维度间全连接，高维数据集可能引入噪声 [^src-crossformer-2023]
2. DLinear（Zeng et al., 2023）在 3/6 数据集上优于所有 Transformer 包括 Crossformer，质疑 Transformer 的排列不变性对时间序列建模的局限 [^src-crossformer-2023]
3. 实验用数据集远小于视觉/文本领域，需要更大更多样的数据 [^src-crossformer-2023]

## 与 wiki 中其他页面的关系

- [[crossformer]] — 实体页
- [[cross-dimension-dependency]] — 跨维度依赖概念
- [[dsw-embedding]] — DSW embedding 技术
- [[two-stage-attention]] — TSA layer 技术
- [[hierarchical-encoder-decoder-ts]] — HED 技术
- [[router-mechanism-for-cross-dimension]] — Router 机制技术
- [[channel-independence]] — CI vs CD 策略对比
- [[patch-based-tokenization]] — 分段嵌入与 patch tokenization 的关系
- [[lstf]] — LSTF 问题设定
- [[cvpe]] — CVPE 借鉴了 Crossformer 的 Router-Attention

[^src-crossformer-2023]: [[source-crossformer-2023]]
