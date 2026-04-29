---
title: "Temporal Query Technique"
type: technique
tags:
  - attention
  - periodicity
  - multivariate
  - time-series
  - global-correlation
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# Temporal Query Technique

Temporal Query (TQ) 技术是一种在注意力机制中融合全局和局部相关性的方法，通过使用**周期性偏移的可学习向量**作为 Query，使模型能够同时捕捉变量间的稳定全局模式和样本级局部变化[^src-tqn]。

## 核心思想

传统自注意力将 Q、K、V 全部从输入序列生成，完全依赖样本级相关性。这种方法在真实世界数据（存在噪声、极端值、缺失值）上容易受到非平稳干扰的影响，导致变量间相关性估计不稳定。

TQ 技术将 Query 从可学习参数生成，而非从输入数据生成，从而在注意力机制中注入全局先验：

- **Query (Q)** = θ_TQ · W_Q，来自周期性偏移的可学习向量 θ_TQ ∈ R^{C×W}
- **Key (K)** = X_t · W_K，来自原始输入序列
- **Value (V)** = X_t · W_V，来自原始输入序列

## 周期偏移机制

对于周期长度 W，提取逻辑为：
```
起始索引 = t mod W
提取长度 = L（look-back window）
```

这意味着每隔 W 个时间步，提取的 TQ 向量相同，实现：
1. **参数复用**：减少参数量
2. **周期对齐**：与数据的内在周期结构对齐
3. **噪声平滑**：通过平均化降低局部噪声干扰

## 注意力计算

```
Head_h = Softmax(Q_h K_h^T / √L) V_h
```

其中 Q_h 来自 TQ 向量，K_h 和 V_h 来自输入数据。这种设计使注意力分数同时反映：
- **全局模式**：TQ 向量捕捉的稳定变量相关性
- **局部样本信息**：输入数据中的样本特异性变化

## 与其他方法的对比

| 方法 | Query 来源 | 特点 |
|------|-----------|------|
| TQ Technique | 可学习向量（周期性偏移） | 融合全局先验 + 局部样本 |
| 标准自注意力 | 输入序列 | 仅样本级相关性 |
| PENGUIN | 输入序列 + 周期偏置 | 在注意力分数中注入周期偏置 |
| CycleNet | 可学习参数（无注意力） | 无注意力机制 |

## 超参数

- **W (周期长度)**：应与数据集的稳定周期对齐，可通过 ACF 或先验知识确定
  - ETTh1/ETTh2: W=24 (日周期)
  - ETTm1/ETTh2: W=96 (日周期，15min 采样)
  - Electricity: W=168 (周周期)
  - Traffic: W=168 (周周期)
  - PEMS: W=288 (日周期，5min 采样)

## 效果

实验表明，融合全局（TQ）和局部（输入）相关性比仅用全局或仅用局部效果更好：

> TQ strategy (global + local) > Sample-only > Global-only[^src-tqn]

## 引用

[^src-tqn]: [[source-tqn]]