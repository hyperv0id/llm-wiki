---
title: "Inverted Transformer Architecture"
type: technique
tags:
  - time-series
  - transformer
  - architecture-design
  - ICLR-2024
created: 2026-05-30
last_updated: 2026-05-31
source_count: 1
confidence: medium
status: active
---

# Inverted Transformer Architecture

Inverted Transformer Architecture 是 iTransformer 提出的架构范式：不修改 Transformer 任何原生组件（attention, FFN, LayerNorm），而是**反转它们的应用维度**——将 attention 作用于变量维度而非时间维度，将 FFN 作用于时间维度而非变量维度[^src-itransformer]。

## 核心公式

```
H^0 = Embedding(X)       # N 个 variate tokens, ℝ^{N×D}
for l = 0, ..., L-1:
    H^l = LayerNorm(H^l + Self-Attention(H^l))   # 变量间交互
    H^l = LayerNorm(H^l + Feed-Forward(H^l))     # 序列表示
Ŷ = Projection(H^L)      # 投影回预测序列
```

## 组件职责重新定义

### Layer Normalization (作用于 variate token)

$$\text{LayerNorm}(\mathbf{H})_n = \frac{h_n - \text{Mean}(h_n)}{\sqrt{\text{Var}(h_n)}}, \quad n = 1, \ldots, N$$

传统做法归一化同一时间步的多变量表示，逐渐融合变量。反转后：
- 归一化每个变量的序列表示，消除不同度量造成的差异
- 所有 variate token 被归一化到高斯分布，减小变量间差异
- 避免传统做法中归一化时间步导致的 oversmooth 问题[^src-itransformer]

### Feed-Forward Network (作用于序列表示)

- 传统：FFN 作用于 temporal token（多变量同时刻），但 token 太局部且变量不对齐
- 反转：FFN 作用于 variate token 的序列表示
- 通用近似定理保证 FFN 可提取复杂序列表示
- FFN 神经元学习时间序列的内禀属性（振幅、周期性、频谱），充当"滤波器"
- 堆叠反转块：编码观测序列 + 解码未来序列表示
- 序列顺序隐式存储在 FFN 神经元排列中 → **无需位置编码**[^src-itransformer]

### Self-Attention (作用于变量间)

- 传统：attention 作用于 temporal tokens 建模时间依赖
- 反转：attention 作用于 variate tokens 建模多变量相关性
- 每个 pre-Softmax 分数 $A_{i,j} = (\mathbf{Q}\mathbf{K}^\top / \sqrt{d_k})_{i,j} \propto q_i^\top k_j$ 揭示变量对之间的相关性
- 整个 score map $\mathbf{A} \in \mathbb{R}^{N \times N}$ 展现多变量相关性结构
- 高相关变量在 representation interaction 中获得更大权重
- 可视化验证：浅层注意力图≈历史变量相关性，深层≈未来变量相关性[^src-itransformer]

## 框架通用性 (iTransformers)

反转架构可直接应用于任何 Transformer 变体，仅改变维度：

| 基础模型 | MSE 提升 (平均) |
|---------|---------------|
| Transformer (2017) | 38.9% |
| Reformer (2020) | 36.1% |
| Informer (2021) | 28.5% |
| Flowformer (2022) | 16.8% |
| Flashformer (2022) | 32.2% |

高效注意力（Reformer/Flowformer/FlashAttention）天然适配变量维度——线性复杂度解决变量数增长的计算瓶颈[^src-itransformer]。

## 高效训练策略

利用变量泛化能力，每 batch 随机采样部分变量训练：

- 20% 采样率下性能轻微下降
- 内存显著减少
- 推理时预测全部变量

## 变量泛化

- Token 数量灵活 → 训练和推理可使用不同变量数
- FFN 学习的序列表示可在不同变量间迁移
- 潜力：支持不同变量数的多变量序列联合训练 → 时序基础模型方向

## 消融验证

| 设计 | Variate 维度 | Temporal 维度 | 性能 |
|------|------------|-------------|------|
| iTransformer | Attention | FFN | **最优** |
| Replace (both attention) | Attention | Attention | 差（Traffic MSE ×2） |
| Vanilla Transformer | FFN | Attention | **最差** |
| w/o Attention | w/o | FFN | 可接受（低维数据） |
| w/o FFN | Attention | w/o | 差 |

## Connections

- [[variate-token-embedding]] — 反转架构的 token 化基础
- [[multivariate-correlation-attention]] — 反转后 attention 的新职责
- [[channel-independence]] — 正交策略：CI 独立变量，iTransformer 保持独立 + attention 关联
- [[itransformer]] — 使用反转架构的代表性模型

[^src-itransformer]: [[source-itransformer]]
