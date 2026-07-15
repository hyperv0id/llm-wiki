---
title: "iTransformer"
type: entity
tags:
  - time-series
  - transformer
  - multivariate-forecasting
  - ICLR-2024
created: 2026-05-30
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

# iTransformer

iTransformer 是清华大学 Mingsheng Long 团队提出的反转 Transformer 架构，发表于 ICLR 2024。其核心思想是不修改 Transformer 任何原生组件，而是将注意力机制和前馈网络的应用维度反转——attention 作用于变量维度捕获多变量相关性，FFN 作用于时间维度学习序列表示[^src-itransformer]。

## 架构概述

```
输入: X ∈ ℝ^{T×N} (T 时间步, N 变量)
1. 转置: X ← X^T  →  ℝ^{N×T}
2. 嵌入: H^0 = MLP(X)  →  ℝ^{N×D}  (每个变量序列 → variate token)
3. L 层 TrmBlock:
   - LayerNorm(H^l) + Self-Attention(H^l)  [变量间交互]
   - LayerNorm(H^l) + Feed-Forward(H^l)     [序列表示]
4. 投影: Ŷ_{:,n} = MLP(h_n^L)  →  ℝ^S
5. 转置: Ŷ ← Ŷ^T  →  ℝ^{S×N}
```

**关键设计选择**：
- **无需位置编码**：序列顺序隐式编码在 FFN 神经元排列中
- **Encoder-only 架构**：预测生成交给线性层（与 DLinear/TiDE 一致）
- **共享 FFN**：所有 variate token 共享同一 FFN，神经元充当通用时间序列"滤波器"

## 与其他模型的定位

| 模型 | Token 化方式 | 多变量相关性 | 时间依赖 |
|------|-------------|-------------|---------|
| Vanilla Transformer | temporal token (多变量同时刻) | 隐式（embedding 融合） | Attention |
| [[crossformer|Crossformer]] | DSW patch token (2D) | 显式（两阶段注意力 + Router） | 两阶段注意力 |
| PatchTST | patch token (CI) | 无（变量独立） | Attention per variate |
| **iTransformer** | **variate token (整条序列)** | **Attention** | **FFN** |

iTransformer 属于论文定义的第四类 Transformer 修改：仅改架构、不改组件。与 [[channel-independence]] (CI) 的区别：CI 完全独立变量但丢失相关性信息，iTransformer 保持变量独立嵌入但通过 attention 显式建模相关性[^src-itransformer]。

## 性能亮点

- **7 个公开数据集**全面 SOTA，高维数据集（Traffic 862 变量、ECL 321 变量）优势最显著
- **框架通用性**：反转后，Transformer +28.5%~38.9% MSE，Reformer +36.1%，Informer +28.5%，Flowformer +16.8%
- **变量泛化**：20% 变量训练即可泛化到全部变量，性能增幅小于 CI-Transformer
- **回看窗口扩展**：随回看窗口增长性能持续提升（传统 Transformer 不提升）
- **高效训练**：随机采样部分变量训练，20% 采样率下内存大幅减少、性能仅轻微下降
- PEMS 数据集 13/13 首位，Market 数据集 28/48 首位

## 消融实验要点

- Vanilla Transformer 排列（attention on temporal + FFN on variate）**表现最差**
- FFN 在时间维度至关重要——CKA 分析证实 iTransformer 学到更高相似度的表示
- 注意力图逐层演化：浅层≈历史变量相关性，深层≈未来变量相关性
- LayerNorm 作用于 variate token：消除度量差异、避免 oversmooth 时间序列

## 局限

1. 单变量场景退化为堆叠线性预测器
2. MLP 嵌入缺乏归纳偏置，对不规则序列鲁棒性待验证
3. 变量数 N 大时 $O(N^2)$ 注意力复杂度需高效注意力插件
4. 未验证大规模预训练潜力

## Connections

- [[channel-independence]] — CI 与 iTransformer 正交：CI 独立变量但丢失相关性，iTransformer 保持独立嵌入 + attention 建模相关性
- [[crossformer]] — Crossformer 显式跨维度依赖但引入 patch 间噪声，iTransformer 更简洁
- [[cross-dimension-dependency]] — iTransformer 提供了建模跨维度依赖的新范式
- [[patch-based-tokenization]] — iTransformer 的 variate token 可视为 Patching 的极端情况（整条序列为一个 patch）
- [[lstf]] — iTransformer 解决了 Transformer 在 LSTF 中随回看窗口增长性能不提升的问题
- [[informer]] — iTransformer 框架可直接提升 Informer 性能 28.5%
- [[granularity-variates]] — SIFusion 将 variate 思路迁移至时间粒度维度（跨粒度 attention）
- [[sifusion]] — SIFusion 中应用 granularity variates 做多粒度海冰预测

[^src-itransformer]: [[source-itransformer]]
