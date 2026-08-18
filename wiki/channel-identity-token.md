---
title: "Channel Identity Tokens (CITs)"
type: technique
tags:
  - time-series
  - transformer
  - channel
  - tokenization
  - representation-learning
created: 2026-08-19
last_updated: 2026-08-19
source_count: 1
confidence: medium
status: active
---

# Channel Identity Tokens (CITs)

Channel Identity Tokens（CITs）是 [[trace|TRACE]] 提出的可学习 token，每通道一个，唯一索引且跨通道不共享，置于该通道 patch 序列首位，充当通道级语义摘要锚点[^src-trace-neurips2025]。

## 设计

- 每通道一个 [CIT]_c ∈ R^(1×d)，从标准高斯分布初始化，与模型联合训练[^src-trace-neurips2025]。
- 在展平的多元 token 序列中，CIT 位于对应通道 patch token 之前：`[CLS]; [CIT]₁; X₁^patch; [CIT]₂; X₂^patch; ...; [CIT]_C; X_C^patch`[^src-trace-neurips2025]。
- CIT 不施加 RoPE（位置无关聚合器），仅聚合本通道信息[^src-trace-neurips2025]。

## 功能

1. **通道解耦表示**：引导模型关注各通道独特行为，学习通道判别式嵌入——传统 decoder-only 基础模型（如 [[timesfm|TimesFM]]、[[chronos|Chronos]]）的嵌入缺乏这种判别力[^src-trace-neurips2025]。
2. **检索锚点**：CIT 嵌入 h_c 在 Stage 2 中与对应通道级文本描述 z_c 进行 channel-level 对齐，实现细粒度跨模态检索[^src-trace-neurips2025]。
3. **连接两阶段**：CIT 是 Stage 1 预训练和 Stage 2 对齐之间的桥梁——Stage 1 学习通道语义摘要，Stage 2 利用它做通道级对比学习[^src-trace-neurips2025]。

## 与其他通道处理策略的关系

- **[[channel-independence|Channel Independence (CI)]]**：CI 完全隔离通道，不建模跨通道依赖；CIT 保留通道内独立处理但通过 [[channel-biased-attention|CbA]] 允许非 CIT token 跨通道交互[^src-trace-neurips2025]。
- **iTransformer variate token**：[[itransformer|iTransformer]] 将整条序列作为一个 variate token，CIT 是更细粒度的通道级摘要[^src-trace-neurips2025]。
- 消融显示移除 CIT 导致 Avg MSE 从 0.670 升至 0.713、Acc 从 85.20% 降至 85.04%[^src-trace-neurips2025]。

## 相关

- [[trace]] — TRACE 模型
- [[channel-biased-attention]] — CbA，与 CIT 配合的注意力机制
- [[channel-independence]] — CI 策略
- [[patch-based-tokenization]] — patch tokenization
- [[dual-level-hard-negative-mining]] — 利用 CIT 做通道级硬负采样

[^src-trace-neurips2025]: [[source-trace-neurips2025]]
