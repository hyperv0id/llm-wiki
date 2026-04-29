---
title: "Gradient Positional Sensitivity (GPS)"
type: technique
tags:
  - position-encoding
  - extrapolation
  - theory
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

**梯度位置敏感性 (Gradient Positional Sensitivity, GPS)** 是无限上下文外推的关键数学条件之一，确保反向传播的梯度携带位置信息，使学习动态能够根据相对位置自适应调整[^src-vetcha-2026-towards-infinite-length-extrapolation]。

## 定义

位置编码方法满足 GPS，当且仅当存在非常数函数 $g(n)$ 使得：

$$\frac{\partial A(n)}{\partial q} = g(n) \cdot \frac{\partial (q^\top k)}{\partial q}$$

即注意力分数关于查询向量的梯度显式依赖于相对位置 $n$[^src-vetcha-2026-towards-infinite-length-extrapolation]。

## 意义

- 确保梯度流携带位置信息
- 使模型能够学习位置特定的适应策略
- 对于训练过程中位置编码的学习至关重要

## 方法对比

| 方法 | GPS | 备注 |
|------|-----|------|
| RoPE | ✅ | 旋转矩阵 $R(n)$ 显式依赖 $n$ |
| ALiBi | ❌ | 偏置 $-m\|n\|$ 与 $q$ 无关，梯度 $\partial A / \partial q = k$ 不含 $n$ |
| APE | ✅ | 旋转矩阵和温度调度都依赖 $n$ |

## ALiBi 缺乏 GPS 的影响

ALiBi 的衰减项 $-m|n|$ 不依赖于 $q$ 或 $k$，因此：

$$\frac{\partial A_{ALiBi}(n)}{\partial q} = \frac{\partial (q^\top k)}{\partial q} = k$$

梯度范数 $\| \partial A / \partial q \| = \|k \|$ 与位置 $n$ 无关，导致模型无法学习位置特定的注意力模式[^src-vetcha-2026-towards-infinite-length-extrapolation]。

---

[^src-vetcha-2026-towards-infinite-length-extrapolation]: [[source-vetcha-2026-towards-infinite-length-extrapolation]]