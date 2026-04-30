---
title: "Gradient Orthogonalization in Optimizers"
type: concept
tags:
  - optimization
  - gradient
  - orthogonalization
created: 2026-04-30
last_updated: 2026-04-30
source_count: 3
confidence: medium
status: active
---

# Gradient Orthogonalization in Optimizers

梯度正交化是一种优化技术，通过对梯度矩阵进行正交变换来改进神经网络训练[^src-muon-optimizer]。

## 背景

在基于动量的优化器（如 SGD-动量、Adam）中，2D 参数的更新矩阵通常具有很高的条件数——即更新矩阵几乎是低秩的，所有神经元的更新都被少数方向主导[^src-muon-optimizer]。

## 正交化的作用

正交化更新矩阵可以：
- 增加"稀有方向"的尺度，这些方向在原始更新中幅度较小但对学习很重要
- 使更新更接近等距变换，避免更新沿单一方向坍缩
- 提高参数空间中的探索效率

## 实现方法比较

| 方法 | 计算复杂度 | 精度要求 | 代表优化器 |
|------|------------|----------|------------|
| SVD | O(n³) | float32 | Orthogonal-SGDM |
| 耦合 Newton 迭代 | O(n²) | float32 | Shampoo |
| Newton-Schulz | O(nm²) | bfloat16 | Muon |

## 历史发展

1. **Stochastic Spectral Descent** (Carlson et al., 2015)：最早使用 SVD 正交化梯度的研究
2. **RMSspectral** (Carlson et al., 2015b)：结合 RMSprop 和随机 SVD
3. **Orthogonal-SGDM** (Tuddenham et al., 2022)：在正交化后应用动量
4. **Muon** (Jordan, 2024)：在动量之后应用正交化，使用 Newton-Schulz

## 关键洞见

Muon 的设计表明：**在正交化之前应用动量**比之前方法中"在正交化之后应用动量"效果更好[^src-muon-optimizer]。

## 参考文献

[^src-muon-optimizer]: [[source-muon-optimizer]]