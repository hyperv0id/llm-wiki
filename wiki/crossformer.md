---
title: "Crossformer"
type: entity
tags:
  - time-series
  - transformer
  - cross-dimension
  - multivariate
  - ICLR-2023
created: 2026-05-30
last_updated: 2026-05-30
source_count: 3
confidence: high
status: active
---

# Crossformer

Crossformer 是首个显式利用跨维度依赖 (cross-dimension dependency) 进行多变量时间序列预测的 Transformer 模型，由 Zhang & Yan 提出（ICLR 2023）[^src-crossformer-2023]。

## 核心架构

Crossformer 由三个组件构成 [^src-crossformer-2023]：

1. **[[dsw-embedding|DSW Embedding]]** — 将输入 MTS 嵌入为 2D 向量阵列（时间 × 维度），保留维度信息
2. **[[two-stage-attention|TSA Layer]]** — 分两阶段捕获跨时间和跨维度依赖，含 [[router-mechanism-for-cross-dimension|Router 机制]] 降低维度间注意力复杂度
3. **[[hierarchical-encoder-decoder-ts|HED]]** — 分层编码器-解码器利用多尺度信息

## 与先前方法的区别

先前 Transformer 模型（Informer、Autoformer、FEDformer 等）将同时间步所有变量嵌入为单一向量，主要建模跨时间依赖，跨维度依赖仅通过 embedding 隐式利用 [^src-crossformer-2023]。Crossformer 的 DSW embedding 反转这一设计：每个向量代表单变量的一段序列，使跨维度依赖可被显式建模。

## 性能

- 6 个数据集、58 个设置中 36 个 top-1、51 个 top-2 [^src-crossformer-2023]
- MTGNN（GNN 建模跨维度依赖）优于多数 Transformer 基线，进一步验证跨维度依赖的重要性 [^src-crossformer-2023]
- 在 ILI 小数据集上不如 FEDformer/Autoformer（引入序列分解先验）[^src-crossformer-2023]
- DLinear 在 3/6 数据集上优于 Crossformer，质疑 Transformer 排列不变性的局限 [^src-crossformer-2023]

## 后续影响

Crossformer 的 Router 机制（路由向量聚合-分发）被 [[cvpe|CVPE]] 借鉴，用于 patch embedding 层的轻量跨变量信息注入 [^src-cvpe-2025]。

## 与 iTransformer 的对比

iTransformer (ICLR 2024) 也显式建模多变量相关性，但采用完全不同的策略[^src-itransformer]：

| 方面 | Crossformer | iTransformer |
|------|------------|-------------|
| Token 化 | DSW 2D patch token (变量 × 时间段) | Variate token (整条变量序列) |
| 跨变量交互 | Cross-Dimension Stage (两阶段注意力 + Router) | Self-attention on variate tokens |
| 时间依赖 | Cross-Time Stage | FFN (共享，等价线性预测器) |
| 组件修改 | 修改注意力 (TSA Layer) | 不修改任何原生组件 |
| 性能 | PEMS 波动序列表现不佳 | PEMS 13/13 首位 |

iTransformer 论文指出 Crossformer 的跨变量 patch 交互引入**时间不对齐噪声**——不同变量的 patch 可能在时间上不对齐，导致注意力图无意义[^src-itransformer]。iTransformer 通过将整条序列嵌入为 token（最大化感受野）避免此问题。

## Connections

- 相关：[[cross-dimension-dependency]] — 核心建模目标
- 相关：[[dsw-embedding]] — DSW embedding
- 相关：[[two-stage-attention]] — TSA layer
- 相关：[[hierarchical-encoder-decoder-ts]] — HED
- 相关：[[router-mechanism-for-cross-dimension]] — Router 机制
- 对比：[[channel-independence]] — CI 策略（不建模跨维度依赖）
- 对比：[[informer]] — 仅建模跨时间依赖
- 对比：[[autoformer]] — 仅建模跨时间依赖
- 后续：[[cvpe]] — 借鉴 Router 机制
- 对比：[[itransformer]] — 不修改组件的反转架构，variety token 交互

[^src-crossformer-2023]: [[source-crossformer-2023]]
[^src-cvpe-2025]: [[source-cvpe-2025]]
[^src-itransformer]: [[source-itransformer]]
