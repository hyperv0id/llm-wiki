---
title: "CVPE"
type: entity
tags:
  - time-series
  - transformer
  - channel-strategy
  - patch-embedding
  - cross-variate
created: 2026-05-30
last_updated: 2026-06-04
source_count: 1
confidence: medium
status: active
---

# CVPE (Cross-Variate Patch Embedding)

CVPE 是一种轻量级模块，将跨变量上下文注入通道独立 (CI) 时间序列预测模型，仅需修改 patch embedding 步骤而不改变后续 CI backbone [^src-cvpe-2025]。

## 设计动机

CI 模型（如 [[patchtst|PatchTST]]、[[time-llm|Time-LLM]]）仅建模时间依赖而忽略跨变量关系，导致模型容量受限；而完全 CD 模型（如 Crossformer、iTransformer）在所有层建模跨变量依赖，容易过拟合噪声。CVPE 在两者之间找到折中——仅在最轻量的 patch embedding 层注入跨变量信息，保留 CI 的鲁棒性 [^src-cvpe-2025]。

## 架构

CVPE 在 vanilla patch embedding 之后引入两个组件 [^src-cvpe-2025]：

1. **可学习位置编码** $W_P \in \mathbb{R}^{P \times d_m}$ — 编码 patch 在时间和变量维度上的相对位置，使后续 router-attention 能感知跨变量上下文
2. **Router-Attention** — 两步 MHA 操作：
   - **聚合**：$A^{(j)} = \text{MHA}_1(R^{(j)}, X_P^{(j)}, X_P^{(j)})$，路由向量 R 作 query 从所有变量聚合信息
   - **分发**：$Z^{(j)} = \text{MHA}_2(X_P^{(j)}, A^{(j)}, A^{(j)})$，将聚合信息分发回各 patch

加上 LayerNorm + MLP 残差连接：
$$\hat{Z} = \text{LayerNorm}(X_P + Z), \quad Z' = \text{LayerNorm}(\hat{Z} + \text{MLP}(\hat{Z}))$$

## 关键特性

| 特性 | 描述 |
|------|------|
| 复杂度 | $O(NP)$，与 CI backbone 对齐 |
| 参数 | 可学习路由向量 $R \in \mathbb{R}^{N \times c \times d_m}$，c 为常数 |
| 即插即用 | 仅替换 patch embedding，无需修改后续层 |
| 适用条件 | 强跨变量相关数据集获益显著；弱相关场景需谨慎 |

## 实验性能

集成到 [[time-llm|Time-LLM]]（GPT-2 backbone, T=256）[^src-cvpe-2025]：

- Weather: 平均 MSE ↓4.6%
- Traffic (Modified): 平均 MSE ↓6.7%
- ETTh1/ETTm1/ECL (Modified): ≈0 性能损失
- ETTh2/ETTm2: ↑5.2% 性能损失（过拟合弱相关特征）

## 局限与未来

- GPU 内存限制：仅评估 GPT-2 backbone（而非 Llama-7B）和修改版 Traffic/ECL [^src-cvpe-2025]
- 弱相关数据集上可能过拟合 [^src-cvpe-2025]
- 未来方向：Channel Partiality — 仅关注部分通道而非全部 [^src-cvpe-2025]

## Connections

- 基于：[[patch-based-tokenization]] — CVPE 修改了 patch embedding 步骤
- 基于：[[router-attention-for-cvpe]] — 核心聚合-分发机制
- 基于：[[learnable-patch-position-encoding]] — 位置编码组件
- 关系：[[channel-independence]] — CVPE 保留 CI backbone 同时注入 CD 信息
- 关系：[[cross-dimension-dependency]] — CVPE 建模的跨维度依赖
- 对比：[[crossformer]] — CVPE Router-Attention 的灵感来源，全 CD 架构
- 对比：[[adaptive-graph-agent-attention]] — 同样用 agent/router token 降低注意力复杂度

[^src-cvpe-2025]: [[source-cvpe-2025]]