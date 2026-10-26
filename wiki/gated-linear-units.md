---
title: "Gated Linear Units (GLU)"
type: technique
tags:
  - neural-network
  - activation-function
  - efficiency
created: 2026-04-29
last_updated: 2026-10-26
source_count: 2
confidence: medium
status: active
---

# Gated Linear Units (GLU)

## 定义

Gated Linear Units (GLU) 是一种神经网络门控机制，结合了线性变换和逐元素门控（element-wise gating），首次在 Language Modeling with Gated Convolutional Networks (2017) 中提出。

## 数学形式

对于输入 Z ∈ R^(N×d)，GLU 定义为：

```
GLU(Z) = σ(ZW_1 + b_1) ⊙ (ZW_2 + b_2)
```

其中：
- σ: Sigmoid 激活函数
- ⊙: 逐元素乘法（element-wise multiplication）
- W_1, W_2: 线性变换矩阵 ∈ R^(d×d)
- b_1, b_2: 偏置向量 ∈ R^d

## 为什么用 GLU 替代 FFN

在 [[mixture-of-experts|MoE]] 架构中，传统做法是使用前馈网络（FFN）作为 experts。但 FFN 存在以下问题：

1. **并行化困难**：多层 FFN 需要复杂机制将 experts 分配到不同计算单元
2. **计算开销高**：每个 expert 都需要独立的前向传播

## FaST 中的 GLU Expert

在 FaST 中，每个 GLU expert 定义为：

```
Exp_i(Z_t^ℓ) = σ(Z_t^ℓ W_{i,2}^ℓ + b_{i,2}^ℓ) ⊙ (Z_t^ℓ W_{i,1}^ℓ + b_{i,1}^ℓ)
```

### 并行化技巧

FaST 通过以下技巧实现高效并行：

```
[Exp_1, ..., Exp_e] = Reshape( σ(F_1) ⊙ F_2 )
其中 F_1, F_2 = Split( Z_t^ℓ W_F + b_F )
W_F ∈ R^(d × 2ed), b_F ∈ R^(2ed)
```

所有 experts 的线性变换被合并成一个线性层，通过 reshape 和 split 操作得到各 expert 输出。

## 实验结果

| 指标 | FFN Expert | GLU Expert |
|------|------------|------------|
| 精度 (MAE) | 19.41 | 19.37 |
| GPU 内存 | 2.5 GB | 1.9 GB |
| 训练速度 | 15s/epoch | 12s/epoch |
| 推理速度 | baseline | **1.4x faster** |

GLU Expert 在保持精度的同时实现 1.4x 推理加速[^src-fast-long-horizon-forecasting]。

## 与其他门控机制的对比

| 机制 | 公式 | 特点 |
|------|------|------|
| LSTM Gate | $\sigma(Wx+b)\odot\tanh(Wx+b)$ | 用于序列建模 |
| GLU | $\sigma(W_1x+b_1)\odot(W_2x+b_2)$ | 双线性变换，信息流量更大 |
| SwiGLU | $\left(\operatorname{Swish}_1(W_1x)\odot(W_2x)\right)W_3$ | 门控激活换成 $\operatorname{Swish}_1$，两升一降，$W_3$ 把宽度降回 $d_{\text{model}}$ |
| 门控残差 | $\sigma(Wx)\odot x+\left(1-\sigma(Wx)\right)\odot F(x)$ | 动态 skip 连接 |

SwiGLU 的两个升维矩阵并列作用于同一输入，逐元素积之后再由 $W_3$ 降维；门控激活是 $\operatorname{Swish}_1$ 而不是 sigmoid，三个矩阵因此是两升一降而非连乘（式 4–6）[^src-shazeer-2020-glu-variants]。

## 优势总结

- **计算高效**：可在一个线性层中并行计算所有 experts
- **保持容量**：保持建模非线性关系的能力
- **梯度流好**：门控机制有助于深层网络训练

## 历史论断

> 截至 2026-05-31 版，SwiGLU 曾记为 $\sigma(W_1x)\odot(W_2x)\odot W_3x$：门控激活写成 sigmoid，三个矩阵排成连乘。
> 已由 [^src-shazeer-2020-glu-variants]（2020-02）的原始定义取代：门控分支先过 $\operatorname{Swish}_1$、内容分支保持线性，两者逐元素相乘后才由 $W_3$ 降维，三个矩阵是两升一降。

## 相关页面

- [[mixture-of-experts|Dense MoE]] — GLU 作为 expert 的应用场景
- [[adaptive-graph-agent-attention|AGA-Att]] — 配合 MoE 的空间注意力模块
- [[swiglu]] — SwiGLU 及 GLU 变体作为 Transformer FFN 的数据流、参数折算与对照结果

[^src-fast-long-horizon-forecasting]: [[source-fast-long-horizon-forecasting]]
[^src-shazeer-2020-glu-variants]: [[source-shazeer-2020-glu-variants]]