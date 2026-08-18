---
title: "Variate Token Embedding"
type: technique
tags:
  - time-series
  - transformer
  - tokenization
  - multivariate
created: 2026-05-30
last_updated: 2026-05-31
source_count: 1
confidence: medium
status: active
---

# Variate Token Embedding

Variate Token Embedding 是 iTransformer 提出的时间序列 token 化方式：将**每个变量的整条时间序列**独立嵌入为一个 token（variate token），而非将同一时间步的多个变量嵌入为一个 temporal token[^src-itransformer]。

## 定义

给定历史观测 $\mathbf{X} = \{x_1, \ldots, x_T\} \in \mathbb{R}^{T \times N}$，其中 $T$ 为时间步数、$N$ 为变量数：

$$h_n^0 = \text{Embedding}(\mathbf{X}_{:,n}) \in \mathbb{R}^D, \quad n = 1, \ldots, N$$

其中 $\text{Embedding}: \mathbb{R}^T \mapsto \mathbb{R}^D$ 由多层感知机 (MLP) 实现。得到 $N$ 个 variate token $\mathbf{H} = \{h_1, \ldots, h_N\} \in \mathbb{R}^{N \times D}$[^src-itransformer]。

## 与传统 Temporal Token 的对比

| 特性 | Temporal Token | Variate Token |
|------|---------------|---------------|
| 构成 | 同一时间步的多变量 | 同一变量的整条序列 |
| 感受野 | 极度局部（单时间步） | 全局（整条序列） |
| 变量独立性 | 融合多变量，丢失独立性 | 保持变量独立 |
| 物理含义 | 混合不同度量/时滞变量 | 同一变量的内禀属性 |
| 位置编码 | 需要 | 不需要（序列顺序隐式于 FFN） |

## 设计动机

1. **时间不对齐**：同一时间步的变量可能不代表同一事件（系统性时滞），如交通传感器在不同区域的事件延迟
2. **度量不一致**：同一时间步的变量可能有不同物理单位和统计分布（如温度 vs 降雨量），嵌入为一个 token 会引入噪声
3. **局部感受野**：单时间步 token 过于局部，无法揭示足够预测信息
4. **与 Patching 的关系**：variate token 可视为 Patching 的极端情况——整条序列视为一个 patch，最大化感受野[^src-itransformer]

## 投影层

预测时，每个 variate token 经投影层恢复为未来序列：

$$\hat{\mathbf{Y}}_{:,n} = \text{Projection}(h_n^L) \in \mathbb{R}^S$$

$\text{Projection}: \mathbb{R}^D \mapsto \mathbb{R}^S$ 同样由 MLP 实现。这与线性预测器（DLinear, TiDE）的设计一致——预测生成交给线性层[^src-itransformer]。

## 扩展方向

- 当前嵌入为简单 MLP，缺乏归纳偏置；可替换为 TCN 等结构化嵌入以处理不规则/不等间距序列
- 嵌入方式支持可变 token 数量，允许训练和推理使用不同变量数

## Connections

- [[patch-based-tokenization]] — Variate token 是 patch token 的极端情况
- [[channel-independence]] — Variate token 保持变量独立嵌入
- [[itransformer]] — 使用 variate token 的代表性模型
- [[crossformer]] — 使用 DSW embedding (2D patch token)

[^src-itransformer]: [[source-itransformer]]
