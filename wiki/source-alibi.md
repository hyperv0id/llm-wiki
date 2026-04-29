---
title: "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
type: source-summary
tags:
  - transformer
  - position-embedding
  - extrapolation
  - iclr-2022
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# ALiBi 论文摘要

## 核心论点

本文提出 **Attention with Linear Biases (ALiBi)** 方法，解决 Transformer 模型在推理时无法处理比训练时更长序列的根本性问题[^src-alibi]。核心发现：位置编码方法是导致外推失败的根本原因[^src-alibi]。

## 主要贡献

1. **首次系统性研究外推问题**：证明 sinusoidal 位置编码在推理时无法处理超过训练长度 20-50 个 token 的序列[^src-alibi]

2. **ALiBi 方法**：不添加位置嵌入，而是在注意力分数上添加与距离成线性关系的偏置惩罚[^src-alibi]

3. **高效外推**：1.3B 参数模型在 L=1024 训练，可外推到 L=2048，性能与在 L=2048 训练的 sinusoidal 模型相当，训练速度快 11%，内存节省 11%[^src-alibi]

## 实验结果

- **WikiText-103**：ALiBi 在 L=512 训练可外推到 12k+ tokens，性能超越 sinusoidal 在 L=3072 的训练结果[^src-alibi]
- **CC100+RoBERTa**：1.3B 参数模型在 L=1024 训练，外推到 L=2048 时 perplexity 优于 sinusoidal 在 L=2048 训练的结果[^src-alibi]
- **效率**：训练速度与 sinusoidal 相当，内存增加可忽略（0-100MB）[^src-alibi]

## 关键发现

- **位置编码决定外推能力**：Sinusoidal、Rotary、T5 Bias 都有外推局限，ALiBi 是首个实现高效外推的方法[^src-alibi]
- **归纳偏置**：ALiBi 对近邻位置有天然偏好（inductive bias towards recency），这与语言模型的实际需求相符[^src-alibi]
- **无需调参**：几何斜率序列 2^(-8/n), 2^(-7/n), ..., 2^0 可泛化到不同模型规模和数据集[^src-alibi]

## 局限性

- 外推性能在约 2L 达到峰值，之后逐渐下降[^src-alibi]
- 在 Traffic 等非语言任务数据集上未验证[^src-alibi]

## 引用

[^src-alibi]: [[source-alibi]]