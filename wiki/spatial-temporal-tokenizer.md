---
title: "Spatial-Temporal Tokenizer"
type: technique
tags:
  - spatial-temporal
  - tokenization
  - plm
  - foundation-model
created: 2026-06-15
last_updated: 2026-06-15
source_count: 1
confidence: high
status: active
---

# Spatial-Temporal Tokenizer

Spatial-Temporal Tokenizer 是 [[std-plm|STD-PLM]] (AAAI 2025) 提出的将时空图数据转换为 PLM 可理解 token 序列的机制，核心创新在于**同时从空间和时间两个维度生成 token**，区别于现有 PLM-based 方法仅从空间维度构造 token 的做法[^src-std-plm]。

## 设计动机

现有 PLM-based 时空模型（STLLM、STGLLM）仅沿空间维度设计 token，忽视了时间维度的全局信息。STD-PLM 认为模型需要同时拥有节点级（微观）和系统级（宏观）的时空理解能力[^src-std-plm]。

## 两类 Token

### 1. Spatial Token（空间 Token）

为每个节点 $n$ 生成一个空间 token $Z_S \in \mathbb{R}^{N \times d_{PLM}}$，由三部分组成[^src-std-plm]：

```
Z_S = LayerNorm(Z_dynamic + Z_intrinsic + Z_mask)
```

| 成分 | 来源 | 含义 |
|------|------|------|
| $Z_{dynamic}$ | MLP(历史数据 $X$) | 节点的微观动态变化 |
| $Z_{intrinsic}$ | MLP([$E_T\|E_N$]) | 节点的静态内在特征（拓扑结构 + 周期性） |
| $Z_{mask}$ | MLP(mask $M$) | 缺失感知——统一预测与插补任务 |

### 2. Temporal Token（时间 Token）

聚合所有节点信息，生成系统级 token[^src-std-plm]：

| Token | 构造方式 | 含义 |
|-------|---------|------|
| $Z_{state}$ | MLP(均值化所有节点的整体状态 $\bar{X}$ + 时间嵌入) | 系统整体状态 |
| $Z_{trend}$ | MLP(一阶差分 $\bar{X}^{trend}$ + 时间嵌入) | 系统整体变化趋势 |

借鉴 [[patchtst|PatchTST]] 的思路，所有时间步合并为一个 patch，仅产生 2 个时间 token——而非为每个时间步单独生成[^src-std-plm]。

## 与现有方法的对比

| 方法 | Token 构造 | 维度 | 拓扑整合 |
|------|-----------|------|---------|
| [[time-llm|Time-LLM]] (ICLR 2024) | 时序 patch + reprogramming | 仅时间 | 无 |
| STLLM (arXiv 2024) | 图节点聚合 | 仅空间 | 邻接矩阵 |
| STGLLM (arXiv 2024) | 图节点 + 时空序列 | 仅空间 | 邻接矩阵 |
| [[nuwats|NuwaTS]] (arXiv 2024) | 统计+缺失+领域嵌入 patch | 仅时间（CI） | 无 |
| **STD-PLM** (AAAI 2025) | **空间 + 时间双维度** | **空间+时间** | **拉普拉斯特征向量** |

## Connections

- 论文：[[std-plm|STD-PLM]] — tokenizer 所属的框架
- 相关：[[topology-aware-node-embedding]] — 空间 token 中 $Z_{intrinsic}$ 所用的节点嵌入
- 相关：[[sandglass-attention]] — token 送入 PLM 前的 SGA 处理
- 概念：[[patch-based-tokenization]] — PatchTST 的 patch tokenization，为时间 token 设计提供参考
- 对比：[[patch-reprogramming]] — Time-LLM 的跨模态对齐式 token

[^src-std-plm]: [[source-std-plm]]
