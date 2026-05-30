---
title: "Router-Attention for CVPE"
type: technique
tags:
  - time-series
  - attention
  - cross-variate
  - router
  - patch-embedding
created: 2026-05-30
last_updated: 2026-05-30
source_count: 1
confidence: medium
status: active
---

# Router-Attention for CVPE

CVPE 中的 Router-Attention 是一种两步聚合-分发注意力机制，借鉴 Crossformer 的路由注意力设计，用于在 patch embedding 层高效注入跨变量信息 [^src-cvpe-2025]。

## 机制

给定位置编码后的 patch embedding $X_P \in \mathbb{R}^{N \times P \times d_m}$ 和可学习路由向量 $R \in \mathbb{R}^{N \times c \times d_m}$（c 为常数），对每个时间步 $j$ 执行两步 MHA [^src-cvpe-2025]：

**第一步 — 聚合**：
$$A^{(j)} = \text{MHA}_1(R^{(j)}, X_P^{(j)}, X_P^{(j)})$$

- R 作 query，$X_P$ 作 key 和 value
- 不学习独立的 Q/K/V 权重矩阵，直接使用 R 和 $X_P$ 作为输入
- c 个路由向量从所有 N 个变量聚合信息，压缩为 $A \in \mathbb{R}^{N \times c \times d_m}$

**第二步 — 分发**：
$$Z^{(j)} = \text{MHA}_2(X_P^{(j)}, A^{(j)}, A^{(j)})$$

- $X_P$ 作 query，A 作 key 和 value
- 将聚合的跨变量信息分发回各 patch，创建 cross-variate-aware embedding

**残差连接**：
$$\hat{Z} = \text{LayerNorm}(X_P + Z), \quad Z' = \text{LayerNorm}(\hat{Z} + \text{MLP}(\hat{Z}))$$

## 复杂度分析

Router-Attention 的复杂度为 $O(NP)$ [^src-cvpe-2025]，因为路由向量数量 c 为常数，不随变量数 N 或 patch 数 P 增长。

## 与 Crossformer Router-Attention 的关系

CVPE 的 Router-Attention 直接借鉴了 Crossformer 的路由注意力概念 [^src-cvpe-2025]，但有两个关键差异：

| 方面 | Crossformer | CVPE |
|------|------------|------|
| 应用层 | 全模型各层 | 仅 patch embedding 层 |
| Backbone | 全 CD 架构 | 保留 CI backbone |
| 目标 | 全局跨维度建模 | 仅 patch 级跨变量注入 |

## 设计优势

- **轻量**：仅 patch embedding 层增加两个 MHA + MLP，不修改后续 LLM 或 Transformer 层 [^src-cvpe-2025]
- **信息保持**：跨变量信息嵌入 patch 后，即使后续层将序列拆分为 N 个独立通道处理，变量间信息仍能通过 patch embedding 传播 [^src-cvpe-2025]

## 相关技术

- 对比：[[adaptive-graph-agent-attention]] — 同样用 agent token 降低 O(N²) → O(Na) 复杂度
- 关系：[[cvpe]] — Router-Attention 是 CVPE 的核心组件
- 关系：[[learnable-patch-position-encoding]] — 位置编码是 Router-Attention 的前置步骤

[^src-cvpe-2025]: [[source-cvpe-2025]]