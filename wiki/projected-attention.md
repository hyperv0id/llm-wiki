---
title: "Projected Attention"
type: technique
tags:
  - attention-mechanism
  - low-rank
  - transformers
  - time-series
created: 2026-05-11
last_updated: 2026-05-11
source_count: 1
confidence: high
status: active
---

# Projected Attention (投影注意力)

**Projected Attention** 是 ImputeFormer 提出的一种低秩约束注意力机制，通过在时间维度上引入可学习投影器将标准自注意力分解为低秩形式，实现线性计算复杂度和显式低秩建模[^src-2312-01728]。

## 动机

时间序列在时域具有冗余性（低秩性），大部分信息可由少数主导模态重建。然而标准自注意力计算出的注意力矩阵可能是高秩的（秩 r ≤ min{T, D'}），既不利于填补不完整隐空间，也 computationally inefficient。同时，稀疏输入上的全注意力易学习虚假相关性。

## 算法

给定投影器 P_proj ∈ R^(C×D') (C < T)，投影注意力分为两步：

### Inflow (信息压缩)

用投影器作为查询，将输入压缩为低维紧凑表示：

```
Z̃_proj = SelfAtten(P_proj, Z, Z) = Softmax(P_proj·W_Q·W_K^T·Z^T / √D') · Z · W_V
```

其中 Z̃_proj ∈ R^(C×D') 存储输入数据中的主要时间模式。

### Outflow (信息恢复)

用原始状态作为查询，投影器作为键字典，恢复完整序列：

```
Ẑ = SelfAtten(Z, P_proj, Z̃_proj) = Softmax(Z·W_Q·W_K^T·P_proj^T / √D') · Z̃_proj · W_V
```

### 整体等效形式

两步合并可展开为：

```
Ẑ ≈ (1/N²) · Q(P^T P)K^T · V
```

其中 Q、K、V 为来自输入的投影。最终注意力矩阵的秩 r ≤ min{C, D'}，低于原始秩 r ≤ min{T, D'}。

## 与线性注意力的比较

与 Linformer 的关键差异：

| 特性 | Linformer | Projected Attention |
|------|-----------|-------------------|
| 压缩矩阵 | 可学习 E,F ∈ R^(C×T)，与数据无关 | 通过 Q·P^T 生成，模式自适应 |
| 低秩形式 | 先算全 QK^T 再压缩 T×T → T×C | 直接分解 T×T → (T×C)(C×T) |
| 参数量 | T×C | C×D' |
| 容量 | 固定静态参数 | 投影器+线性层，更大的模型容量 |

## 复杂度

O(TC) 线性于序列长度（标准自注意力为 O(T²)）。

## 可解释性

- Inflow 注意力图量化了不完整信息流如何被压缩到低维空间
- 不同注意力头提供不同密度的信息
- Outflow 显示少数时间模态即可重建有效神经表征
- 经过投影注意力层后，隐状态的奇异值分布更接近完整数据的低秩分布

## 关联页面

- [[imputeformer]] — ImputeFormer 模型
- [[embedded-attention]] — 空间嵌入注意力
- [[fourier-imputation-loss]] — 傅里叶填补损失

[^src-2312-01728]: [[source-2312-01728]]
