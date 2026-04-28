---
title: "Position Extrapolation"
type: technique
tags:
  - transformer
  - position-encoding
  - extrapolation
created: 2026-04-28
last_updated: 2026-04-28
source_count: 2
confidence: high
status: active
---

# Position Extrapolation

位置外推（Position Extrapolation）是指 Transformer 模型在推理时处理比训练时更长序列的能力[^src-alibi]。

## 问题定义

- **训练长度 $L$**：训练时每个输入子序列的 token 数[^src-alibi]
- **推理长度 $L_{valid}$**：验证/推理时的输入序列长度[^src-alibi]
- **外推**：当 $L_{valid} > L$ 时，模型仍能保持或提升性能[^src-alibi]

## 为什么外推重要

1. **训练效率**：长序列训练成本高（注意力矩阵 $O(L^2)$）[^src-alibi]
2. **内存限制**：长序列需要大量 GPU 内存[^src-alibi]
3. **实际需求**：推理时往往需要处理比训练时更长的上下文[^src-alibi]

## 不同位置编码的外推能力

| 位置编码 | 外推能力 | 原因 |
|---------|---------|------|
| Learned | 无 | 无法编码超过 L 的位置[^src-alibi] |
| Sinusoidal | 极弱（~L+50） | 理论可外推，实际失败[^src-alibi] |
| Rotary | 弱（~L+200） | 部分外推，有额外开销[^src-alibi] |
| T5 Bias | 中等（~2L） | 外推好但训练慢 2 倍[^src-alibi] |
| **ALiBi** | **强（>3L）** | 线性偏置天然支持外推[^src-alibi] |
| **APE** | **极强（>256L）** | 理论支持无限上下文外推[^src-vetcha-2026-towards-infinite-length-extrapolation] |

## ALiBi 的外推机制

ALiBi 使用线性偏置而非绝对位置编码，因此[^src-alibi]：

1. **无位置上限**：偏置仅依赖相对距离，不依赖绝对位置
2. **归纳偏置**：对近邻的偏好是通用的，不随序列长度改变
3. **平滑衰减**：线性偏置使远距离 token 的影响平滑衰减

## 外推性能曲线

ALiBi 的外推性能呈现以下特征[^src-alibi]：

- **上升阶段**（$L_{valid} \in [L, 2L]$）：性能持续提升
- **峰值**（$L_{valid} \approx 2L$）：达到最佳 perplexity
- **平稳阶段**（$L_{valid} \in [2L, 3L]$）：性能保持稳定
- **缓慢下降**（$L_{valid} > 3L$）：性能逐渐下降但仍可用

## 实际应用

- **Few-shot prompting**：可输入更多示例到上下文[^src-alibi]
- **长文档处理**：处理超过训练长度的文档[^src-alibi]
- **长文本生成**：生成比训练样本更长的输出[^src-alibi]

## 引用

[^src-alibi]: [[source-alibi]]
[^src-vetcha-2026-towards-infinite-length-extrapolation]: [[source-vetcha-2026-towards-infinite-length-extrapolation]]