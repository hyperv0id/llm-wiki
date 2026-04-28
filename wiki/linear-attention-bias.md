---
title: "Linear Attention Bias"
type: technique
tags:
  - transformer
  - attention
  - position-encoding
created: 2026-04-28
last_updated: 2026-04-28
source_count: 2
confidence: high
status: active
---

# Linear Attention Bias

线性注意力偏置是 ALiBi 的核心技术，通过在注意力分数上添加与距离成线性关系的偏置来编码位置信息[^src-alibi]。

## 数学定义

对于第 $i$ 个 query $q_i$，注意力计算变为[^src-alibi]：

$$\text{softmax}(q_i K^T + m \cdot [-(i-1), ..., -2, -1, 0])$$

其中：
- $m$ 是每个注意力头的斜率（slope），为固定值[^src-alibi]
- 偏置向量 $[-(i-1), ..., -2, -1, 0]$ 表示 query 与每个 key 的距离的负数[^src-alibi]

## 工作原理

1. **距离惩罚**：query 与 key 距离越远，偏置越负，注意力分数越低[^src-alibi]
2. **近邻偏好**：模型天然倾向于关注近邻 token，符合语言模型的 recency 需求[^src-alibi]
3. **多头差异化**：不同头使用不同斜率，捕捉不同距离范围的依赖关系[^src-alibi]

## 实现方式

ALiBi 通过修改注意力掩码矩阵实现，只需几行代码[^src-alibi]：

```python
# 为每个头生成线性偏置
bias = torch.arange(L).unsqueeze(0) - torch.arange(L).unsqueeze(1)
bias = -bias.float() * slope  # 负偏置，距离越远越小
attention_scores = attention_scores + bias
```

## 与其他位置编码的区别

| 方法 | 位置信息注入方式 | 是否可学习 |
|------|-----------------|-----------|
| Sinusoidal | 加到词嵌入 | 否 |
| Learned | 加到词嵌入 | 是 |
| Rotary | 乘到 Q 和 K | 否 |
| T5 Bias | 加到注意力分数 | 是 |
| **ALiBi** | **加到注意力分数** | **否** |

## 优势

- **无额外运行时开销**：偏置在掩码矩阵中预计算[^src-alibi]
- **无额外参数**：斜率固定，不参与训练[^src-alibi]
- **内存增加可忽略**：0-100MB（因多头掩码矩阵略大）[^src-alibi]
- **外推能力**：训练短序列，推理长序列[^src-alibi]

## 局限性

- 线性衰减导致远距离 token 的注意力分数被完全抑制，无法保持长程依赖性（缺乏 [[long-distance-correlation-preservation|LDCP]]）[^src-vetcha-2026-towards-infinite-length-extrapolation]
- 偏置与查询向量无关，导致梯度不携带位置信息（缺乏 [[gradient-positional-sensitivity|GPS]]）[^src-vetcha-2026-towards-infinite-length-extrapolation]
- [[adaptive-positional-encoding|APE]] 通过次线性衰减（log + √|n|）改进此问题[^src-vetcha-2026-towards-infinite-length-extrapolation]

## 引用

[^src-alibi]: [[source-alibi]]
[^src-vetcha-2026-towards-infinite-length-extrapolation]: [[source-vetcha-2026-towards-infinite-length-extrapolation]]