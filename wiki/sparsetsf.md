---
title: "SparseTSF"
type: entity
tags:
  - time-series
  - forecasting
  - model
  - lightweight
  - tpami-2026
  - icml-2024
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# SparseTSF

SparseTSF 是 IEEE TPAMI 2026 接收的极轻量级时序预测模型，由华南理工大学林伟伟团队提出（ICML 2024 Oral）。其核心创新是 **Cross-Period Sparse Forecasting** 技术，通过跨周期稀疏预测将模型参数量降至 **1,000 以下**[^src-sparsetsf]。

## 核心创新

### Cross-Period Sparse Forecasting

传统 LTSF 模型需要 O(L×H) 的参数量来建模长时间依赖。SparseTSF 将这一任务分解为：

1. **下采样**：将长度为 L 的序列按周期 w 重排为 w × (L/w) 的矩阵
2. **共享预测**：用单一 Linear/MLP backbone 预测所有子序列
3. **上采样**：将预测结果重排回长度为 H 的序列

参数量从 O(L×H) 降至 O(L/w × H/w)，实现 1~4 个数量级的压缩。

### 隐式正则化

论文从理论上证明：Sparse 技术等价于在权重矩阵上施加结构化约束——惩罚非周期位置的连接（稀疏），强制周期内参数一致（共享）。这起到类似 L1 正则化的作用，增强模型对噪声的鲁棒性。

## 架构变体

| 变体 | 参数量 | 适用场景 |
|------|--------|----------|
| SparseTSF/Linear | < 1,000 | 单变量预测、计算资源极度受限 |
| SparseTSF/MLP | < 8,000 | 高维多变量预测 |

## 性能

- 在 ETTh1, ETTh2, Electricity, Traffic, Solar-Energy, Weather 上取得 SOTA 或接近 SOTA
- 相比 FITS（10k 参数），SparseTSF 参数量再降 10 倍
- 长 look-back 窗口（720）下优势更明显

## 与其他轻量模型的关系

| 模型 | 参数量 | 技术路线 |
|------|--------|----------|
| DLinear | ~1k | 单层线性 |
| FITS | ~10k | 频域 + 低通滤波 |
| **SparseTSF** | **<1k** | **跨周期稀疏预测** |
| CycleNet | ~10k | 可学习周期参数 |

## 引用

[^src-sparsetsf]: [[source-sparsetsf]]