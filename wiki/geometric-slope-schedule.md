---
title: "Geometric Slope Schedule"
type: technique
tags:
  - transformer
  - attention
  - hyperparameter
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# Geometric Slope Schedule

几何斜率调度是 ALiBi 中为多头注意力分配不同斜率的方法，使不同头关注不同距离范围的依赖关系[^src-alibi]。

## 定义

对于有 $n$ 个注意力头的模型，斜率 $m$ 采用几何序列[^src-alibi]：

$$m_i = 2^{-8i/n}, \quad i = 1, 2, ..., n$$

即：$2^{-8/n}, 2^{-7/n}, ..., 2^0$

## 具体示例

### 8 头模型

$$m = [2^{-1}, 2^{-2}, 2^{-3}, 2^{-4}, 2^{-5}, 2^{-6}, 2^{-7}, 2^{-8}]$$

即：$[0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]$

### 16 头模型

通过对 8 头斜率相邻对几何平均插值得到[^src-alibi]：

$$m = [2^{-0.5}, 2^{-1}, 2^{-1.5}, ..., 2^{-8}]$$

## 设计原理

1. **斜率范围 (0, 1]**：所有斜率在 0 到 1 之间[^src-alibi]
2. **密度递增**：越接近 0，斜率越密集（几何序列的特性）[^src-alibi]
3. **小斜率 = 长距离**：斜率越小，偏置增长越慢，允许关注更远距离[^src-alibi]
4. **大斜率 = 短距离**：斜率越大，偏置增长越快，强制关注近邻[^src-alibi]

## 泛化性

该斜率调度具有极强的泛化能力[^src-alibi]：

- **跨数据集**：WikiText-103（维基百科）和 Toronto Book Corpus（书籍）使用相同斜率[^src-alibi]
- **跨模型规模**：从 16 层到 25 层，从 100M 到 1.3B 参数[^src-alibi]
- **跨训练预算**：从 205 epoch 到 1 epoch（50k updates）[^src-alibi]

## 消融实验

- **可学习斜率**：尝试让斜率参与训练，但外推效果不佳[^src-alibi]
- **随机采样**：从指数分布随机采样斜率，部分有效但方差大[^src-alibi]
- **手动探索**：约 10 组斜率的手动探索发现了当前最优方案[^src-alibi]

## 引用

[^src-alibi]: [[source-alibi]]