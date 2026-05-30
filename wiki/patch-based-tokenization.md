---
title: "Patch-based Tokenization"
type: technique
tags:
  - time-series
  - transformer
  - tokenization
  - patch
created: 2026-04-28
last_updated: 2026-05-30
source_count: 2
confidence: medium
status: active
---

# Patch-based Tokenization

Patch-based tokenization 是 SimDiff 等现代时间序列 Transformer 使用的关键预处理技术，将连续的时间序列转换为离散的 token 序列 [^src-simdiff]。

## 方法

1. **Patch 划分**：将时间序列划分为重叠的固定长度窗口（patches）
2. **Token 映射**：每个 patch 通过密集 MLP 转换为 token embedding
3. **时间步嵌入**：扩散时间步 k 被处理为时间 token，与原始 tokens 拼接

## 优势

- **局部依赖建模**：每个 patch 作为 token 捕获局部依赖关系 [^src-simdiff]
- **计算效率**：减少序列长度，降低注意力计算成本
- **信息聚合**：patch 内信息自动聚合，减少噪声

## 在 SimDiff 中的应用

SimDiff 使用 patch-based tokenization 将时间序列转换为重叠的 tokens，然后通过 Transformer backbone 进行去噪处理 [^src-simdiff]。这种设计使模型能够平衡简洁性和深度，确保鲁棒且高效的时间序列预测 [^src-simdiff]。

## 跨变量增强：CVPE

CVPE (Cross-Variate Patch Embedding) 在 vanilla patch embedding 后添加可学习位置编码 $W_P \in \mathbb{R}^{P \times d_m}$ 和 Router-Attention 机制，使 patch token 在被送入后续 CI 层之前就已携带跨变量信息 [^src-cvpe-2025]。这意味着即使后续层将序列拆分为 N 个独立通道处理，跨变量上下文仍能通过 patch embedding 传播 [^src-cvpe-2025]。实验证明：在 Weather 和 Traffic 等强相关数据集上显著提升（↓4.6%-6.7% MSE），但在弱相关数据集上可能过拟合 [^src-cvpe-2025]。

## 相关技术

- 对比：[[channel-independence]] — 通道独立处理
- 对比：[[instance-normalization]] — RevIN 策略
- 相关：[[normalization-independence]] — SimDiff 的归一化技术
- 相关：[[cvpe]] — 跨变量增强的 patch embedding
- 相关：[[router-attention-for-cvpe]] — CVPE 的聚合-分发注意力机制
- 相关：[[learnable-patch-position-encoding]] — CVPE 的可学习位置编码

[^src-simdiff]: [[source-simdiff]]
[^src-cvpe-2025]: [[source-cvpe-2025]]