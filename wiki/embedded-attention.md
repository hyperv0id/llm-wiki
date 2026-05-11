---
title: "Embedded Attention"
type: technique
tags:
  - attention-mechanism
  - spatial
  - low-rank
  - transformers
created: 2026-05-11
last_updated: 2026-05-11
source_count: 1
confidence: high
status: active
---

# Embedded Attention (嵌入注意力)

**Embedded Attention** 是 ImputeFormer 提出的一种低秩空间注意力机制，通过节点嵌入作为低维代理计算传感器间的相关性图，在不依赖预定义图结构的前提下实现线性复杂度 O(N·D_emb)[^src-2312-01728]。

## 动机

对传感器间空间依赖关系的建模是时空填补的关键。传统方法依赖预定义图（如物理距离图）或在全量数据上计算全局空间注意力（复杂度 O(N²·T·D')）。前者需要先验知识，后者存在高秩和高计算成本问题。缺失数据的存在又进一步加剧了注意力图中的虚假相关性。

## 方法

### 节点嵌入

每个传感器 s_i 关联一个可学习参数 e_i ∈ R^(D_s)，作为该传感器的稠密低维抽象。嵌入经过多头顶展开后形成时变表示 E^(i)_(t:t+T) ∈ R^(T×D_s/T)。静态节点嵌入取时间头顶的平均值：E = [ē_1‖ē_2‖···‖ē_N] ∈ R^(N×D_s/T)。

### 注意力计算

给定静态节点嵌入 E，通过线性投影生成查询和键：

```
Q_e^(ℓ) = Linear(E), K_e^(ℓ) = Linear(E)
A^(ℓ) = Softmax(Q_e^(ℓ)·K_e^(ℓ)^T / √D')
```

其中 A^(ℓ) ∈ R^(N×N) 表示 N 个传感器间的成对相关分数。

### 线性化技巧

利用矩阵乘法的结合律和分离 softmax 实现线性复杂度：

```
A^(ℓ) ≈ σ₂(Q̃_e^(ℓ)) · σ₁(K̃_e^(ℓ))^T
```

其中 Q̃_e^(ℓ)、K̃_e^(ℓ) 为 ℓ₂ 归一化版本。完整的嵌入注意力层：

```
Z̃_t^(ℓ) = LayerNorm(Z_t^(ℓ) + σ₂(Q̃_e^(ℓ))·σ₁(K̃_e^(ℓ))^T·Z_t^(ℓ))
Z_t^(ℓ+1) = LayerNorm(Z̃_t^(ℓ) + FeedForward(Z̃_t^(ℓ)))
```

### 低秩性质

嵌入注意力生成注意力图的秩 r ≤ min{N, D_emb}。由于 D_emb ≪ D'（模型维度），注意力图的秩显著低于全空间注意力的秩 r ≤ min{N, T·D'}。

## 复杂度

- 全空间注意力：O(N²·T·D')
- 嵌入注意力：O(N·D_emb) — 线性于传感器数量

## 优势

- 不依赖预定义图结构，通过端到端学习自动推断空间关系
- 对缺失值鲁棒：节点嵌入与时间信息解耦，缺失数据不影响相关性推断
- 低秩特性有效过滤噪声和虚假相关性
- 可解释：t-SNE 可视化显示邻近传感器形成聚类，与实际道路拓扑一致

## 关联页面

- [[imputeformer]] — ImputeFormer 模型
- [[projected-attention]] — 时间投影注意力
- [[fourier-imputation-loss]] — 傅里叶填补损失

[^src-2312-01728]: [[source-2312-01728]]
