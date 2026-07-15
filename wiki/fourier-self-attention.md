---
title: "Fourier Self-Attention"
type: technique
tags:
  - fourier-transform
  - self-attention
  - frequency-domain
  - weather-forecasting
  - transformer
created: 2026-07-14
last_updated: 2026-07-15
source_count: 1
confidence: medium
status: active
---

# Fourier Self-Attention

**傅里叶自注意力（Fourier Self-Attention）** 是 [[cirt|CirT]] 提出的一种注意力机制：在 Transformer 每层内对 patch embedding 做 DFT → 频域多头注意力 → IDFT，以显式编码空间周期性[^src-cirt]。

## 操作流程

在第 $l$ 层 Transformer block 中[^src-cirt]：

1. **DFT**：对每个 patch 的 embedding $E_h^{(l)} \in \mathbb{R}^D$ 做离散傅里叶变换：
   $$S_h^{(l)} = \mathcal{F}(E_h^{(l)}) = A_h^{(l)} - B_h^{(l)}i$$
   其中 $A_h^{(l)} = \text{Re}(S_h^{(l)})$，$B_h^{(l)} = \text{Im}(S_h^{(l)})$

2. **频域拼接**：所有 patch 的实虚部堆叠为 $C^{(l)} = [A^{(l)}, B^{(l)}] \in \mathbb{R}^{H \times 2D}$

3. **多头注意力**：标准 scaled dot-product attention
   $$C^{(l,m)} = \text{softmax}\left(\frac{Q^{(l,m)}K^{(l,m)\top}}{\sqrt{D}}\right)V^{(l,m)}$$

4. **实虚部分离**：$\tilde{A}^{(l,m)}$ 取前 $D$ 列，$\tilde{B}^{(l,m)}$ 取后 $D$ 列

5. **IDFT**：重构空间域表示
   $$\tilde{E}_{h,n}^{(l,m)} = \frac{1}{D}\sum_{k=1}^D \left(\tilde{A}_{h,k}^{(l,m)}\cos(2\pi k \frac{n}{N}) - \tilde{B}_{h,k}^{(l,m)}\sin(2\pi k \frac{n}{N})\right)$$

6. **FFN**：拼接所有 head 的输出后通过 MLP

## 与 FEDformer 频域注意力的区别

FEDformer (Zhou et al., 2022) 同样在 Transformer 中利用傅里叶变换处理序列周期性，两者均认同频域操作对周期信号建模的价值[^src-cirt]。但路径不同：FEDformer 在频域使用随机频率子集配合可学习核做 element-wise 处理，旨在降低长序列的计算开销；CirT 则保留完整频率表示并在实部/虚部拼接后执行标准 scaled dot-product multi-head attention——这一选择源自 [[circular-patching|圆形分块]] 的空间周期性的严格建模需求：circular patch 的 $2\pi$ 周期性使得其 DFT 系数构成完备基函数表示，标准 attention 可在此空间中学习各频率分量间的全局交互[^src-cirt]。

## 设计原理

CirT 的设计源自 [[circular-patching|圆形分块]] 的 $2\pi$ 周期性：circular patch 的傅里叶系数天然构成一组周期性基函数的完备表示。在频域做 attention 等价于在这些基函数系数上学习全局交互，相比时域注意力能更高效地捕获 latitudinal 方向的长程周期依赖[^src-cirt]。

## 相关页面

- [[cirt]] — CirT 模型
- [[circular-patching]] — 圆形分块
- [[frequency-enhanced-block]] — FEDformer 的 FEB
- [[frequency-enhanced-attention]] — FEDformer 的 FEA
- [[spherical-geometry-inductive-bias]] — 球面几何归纳偏置
- [[source-cirt]] — CirT 论文

[^src-cirt]: [[source-cirt]]