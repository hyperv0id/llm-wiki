---
title: "Patch-based Tokenization"
type: technique
tags:
  - time-series
  - transformer
  - tokenization
  - patch
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
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

## 相关技术

- 对比：[[channel-independence]] — 通道独立处理
- 对比：[[instance-normalization]] — RevIN 策略
- 相关：[[normalization-independence]] — SimDiff 的归一化技术

[^src-simdiff]: [[source-simdiff]]