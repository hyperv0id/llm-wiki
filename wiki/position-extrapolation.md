---
title: "Position Extrapolation"
type: technique
tags:
  - transformer
  - position-encoding
  - extrapolation
created: 2026-04-28
last_updated: 2026-10-26
source_count: 2
confidence: high
status: active
---

# Position Extrapolation

位置外推（Position Extrapolation）是指 Transformer 模型在推理时处理比训练时更长序列的能力[^src-alibi]。

## 问题定义

训练长度 $L$ 为训练时每个输入子序列的 token 数[^src-alibi]，推理长度 $L_{valid}$ 为验证或推理时的输入序列长度[^src-alibi]；外推指当 $L_{valid} > L$ 时，模型仍能保持或提升性能[^src-alibi]。

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
| **ALiBi** | **强（$\ge 3L$；$L=512$ 到 12k 仍改善）** | 加性线性偏置，无绝对位置、无按距离查表[^src-alibi] |
| **APE** | **极强（>256L）** | 理论支持无限上下文外推[^src-vetcha-2026-towards-infinite-length-extrapolation] |

## ALiBi 的外推机制

ALiBi 使用线性偏置而非绝对位置编码，因此[^src-alibi]：

1. **无位置上限**：偏置仅依赖相对距离，不依赖绝对位置
2. **归纳偏置**：对近邻的偏好是通用的，不随序列长度改变
3. **平滑衰减**：线性偏置使远距离 token 的影响平滑衰减

## 外推性能曲线

曲线形状随数据集与评估方式变化，两套设置得出的结论不同[^src-alibi]：

WikiText-103 非重叠评估中，$L=512$ 模型从 19.73（$L_{valid}=512$）改善到 18.40（3072），15512 token 时仍为 18.31；$L=3072$ 模型从 17.60 到 17.15（13072）、17.22（16072），全程未退化（Table 2、Table 4）。CC100+RoBERTa（1.3B）上，$L=512$ 与 $L=1024$ 模型都在约 $2L$ 处取到最好值（1012 时 9.3、2024 时 8.9），随后困惑度缓慢回升，到 10000 token 仍保持较强水平（Figure 6）；推理子序列长为 $2L$ 时仍有一半预测落在训练长度内，超过 $2L$ 后该比例跌破一半（§4.2）。

## 收益来源的边界

改用 stride $S=1$ 的滑窗评估并让每个预测获得最长可用上下文后，ALiBi 的困惑度随 $L_{valid}$ 基本持平：$L=512$ 从 17.98 到 18.30，$L=1024$ 从 17.46 到 17.92（Table 15）。非重叠评估下的改善主要来自 early token curse 的缓解——更长子序列降低了开局缺少上下文的预测占比，而非用上了超过 $L$ 的上下文（Appendix B）[^src-alibi]。

## 实际应用

模型通过训练短序列、推理时直接输入更长子序列来节省显存与训练时间[^src-alibi]。这免去了长序列预训练：$L=1024$ 训练即可覆盖 2048 输入长度；训练同样时长下 $L_{valid}=2048$ 时为 8.92，直接训练 $L=2048$ 的 sinusoidal 为 9.01，显存 26.2 对 29.3 GB（§4.2、Table 11）[^src-alibi]。同时，推理时可在上下文里放入更多训练示例（摘要）[^src-alibi]，并支持生成长于训练子序列的输出（摘要）[^src-alibi]。

## 引用

[^src-alibi]: [[source-alibi]]
[^src-vetcha-2026-towards-infinite-length-extrapolation]: [[source-vetcha-2026-towards-infinite-length-extrapolation]]
