---
title: "Channel-biased Attention (CbA)"
type: technique
tags:
  - time-series
  - transformer
  - attention
  - channel
  - multivariate
created: 2026-08-19
last_updated: 2026-08-19
source_count: 1
confidence: medium
status: active
---

# Channel-biased Attention (CbA)

Channel-biased Attention（CbA）是 [[trace|TRACE]] 提出的注意力机制，通过偏置注意力掩码 M ∈ {0,1}^{L×L} 在多通道时序 Transformer 中实现通道解耦的归纳偏置，同时保留跨通道信息流动[^src-trace-neurips2025]。

## 机制

- 对于位于展平序列索引 i_c 的 [[channel-identity-token|CIT]]_c：M_{i_c, j} = 0 当 token j 不属于通道 c，否则为 1——CIT 仅关注本通道 token[^src-trace-neurips2025]。
- 对于非 CIT token k：M_{k, j} = 1——非 CIT token 可自由关注全序列所有 token[^src-trace-neurips2025]。
- RoPE 在 Q/K 上施加，仅在每通道内的 T̂ 个时间 token 上独立旋转，使用原始时间差 Δt_ij 而非展平序号[^src-trace-neurips2025]。

注意力公式：α_ij = softmax_j(Q^T R_{θΔt_ij} K / √d + log M_{ij})[^src-trace-neurips2025]。

## 设计意图

CbA 在 CI 与 CD 之间提供折中：

| 策略 | CIT 关注范围 | 非 CIT 关注范围 | 跨通道交互 |
|------|-------------|----------------|------------|
| [[channel-independence|CI]] | N/A（通道完全隔离） | N/A | ✗ |
| **CbA** | **仅本通道** | **全序列** | **有限（通过非 CIT token）** |
| Full Attention | 全序列 | 全序列 | ✓（无约束） |

## 消融

- CbA → Full Attention：Avg MSE 0.670→0.713，Acc 85.20→84.18[^src-trace-neurips2025]。
- CbA → Causal Attention：Avg MSE 0.670→0.705，Acc 85.20→83.72[^src-trace-neurips2025]。

CbA + CIT 协同效果最优[^src-trace-neurips2025]。

## 相关

- [[trace]] — TRACE 模型
- [[channel-identity-token]] — CIT，与 CbA 配合
- [[channel-independence]] — CI 策略（CbA 的折中对象）
- [[roformer]] — RoPE（CbA 内使用）

[^src-trace-neurips2025]: [[source-trace-neurips2025]]
