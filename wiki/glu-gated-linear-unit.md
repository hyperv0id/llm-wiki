---
title: "Gated Linear Unit (GLU)"
type: technique
tags:
  - attention
  - feed-forward
  - activation-function
created: 2026-05-04
last_updated: 2026-10-26
source_count: 2
confidence: medium
status: active
---

# Gated Linear Unit (GLU)

**Gated Linear Units (GLU)** are activation-modulated linear transformations where the output is a component-wise product of two linear projections, one of which is gated by a non-linear activation:

$$\text{GLU}(\mathbf{x}) = (\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1) \odot \sigma(\mathbf{x}\mathbf{W}_2 + \mathbf{b}_2)$$

GLU variants (SwiGLU, GEGLU, etc.) replace the sigmoid $\sigma$ with other activations like Swish or GELU. GLUs have become the dominant choice for feed-forward layers in modern Transformer architectures (LLaMA, PaLM, etc.) and are also used as efficient MoE experts in models like [[mixture-of-experts|FaST]].

## Transformer FFN 中的无偏置形式与变体

上面的式子含偏置，是 Dauphin 等人的原定义，Shazeer 复述为式 4。在 T5 代码库的无偏置约定下，它被写成 $\operatorname{GLU}(x,W,V)=\sigma(xW)\otimes xV$，再接第三个矩阵 $W_{\text{out}}$ 降回 $d_{\text{model}}$，即三矩阵 FFN；门控激活换成 ReLU、GELU、$\operatorname{Swish}_1$ 与恒等分别得到 ReGLU、GEGLU、SwiGLU 与 Bilinear（式 6）[^src-shazeer-2020-glu-variants]。三矩阵对两矩阵按 $d_{ff}\to\tfrac{2}{3}d_{ff}$ 折算（T5-base 3072→2048）。在同一套 T5-base 文本迁移设置下，五个门控变体的 heldout log-perplexity 全部低于 ReLU/GELU/Swish 基线，下游均分多数更高，SQuAD 上则 GLU 低于 ReLU（表 1–4）[^src-shazeer-2020-glu-variants]。逐阶段数据流、宽度折算与逐表数字见 [[swiglu]]。

## 与 Mamba 输入门的关系

Mamba 的输入门 $\mathbf{\Delta}_i$ 本质上是一种 GLU 变体——它对输入 $\mathbf{x}_i$ 进行逐元素门控后再投影到隐藏状态[^src-demystify-mamba-linear-attention-2024]。具体而言：

$$\mathbf{\Delta}_i = \text{Softplus}(\mathbf{x}_i W_1 W_2)$$

这与 GLU 的 $\sigma(\mathbf{x}\mathbf{W}_2 + \mathbf{b}_2) \odot (\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)$ 模式一致，区别在于 Mamba 使用 Softplus 而非 sigmoid 作为门控激活函数，且门控发生在投影之前而非之后。消融实验表明输入门贡献了 +0.9% 的 ImageNet-1K 准确率提升[^src-demystify-mamba-linear-attention-2024]。

## Related Pages
- [[mixture-of-experts]]
- [[gated-linear-units]]
- [[swiglu]] — SwiGLU 及 GLU 变体 FFN 的数据流、参数折算与 T5-base 对照结果
- [[gated-residual-network]] — TFT 中以 GLU 门控残差非线性分支的构建块
- [[tft]] / [[source-tft]] — 时序预测中广泛使用 GRN+GLU 的代表模型

[^src-demystify-mamba-linear-attention-2024]: [[source-demystify-mamba-linear-attention-2024]]
[^src-shazeer-2020-glu-variants]: [[source-shazeer-2020-glu-variants]]