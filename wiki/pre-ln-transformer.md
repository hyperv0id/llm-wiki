---
title: "Pre-LN Transformer（前置层归一化）"
type: technique
tags:
  - transformer
  - layer-normalization
  - residual-connection
  - architecture
  - language-model
created: 2026-10-26
last_updated: 2026-10-27
source_count: 2
confidence: medium
status: active
---

# Pre-LN Transformer（前置层归一化）

Pre-LN 指把 LayerNorm 放在每个子层输入端、残差相加之前的 Transformer 块排布，与 Transformer 原文在残差相加之后的 post-LN 相对。GPT-1 的 12 层 decoder 仍是 post-LN 排布，未做此改动（§4.1）[^src-radford-2018-gpt]。GPT-2 报告在沿用 GPT 时列出这一改动：LayerNorm 移到每个子块输入端，并在最后一个 self-attention block 之后补一层 LayerNorm（§2.3）[^src-gpt2-2019-language-models]。报告用「与 pre-activation 残差网络相似」（similar to）描述该排布，未使用 pre-LN 这一名称[^src-gpt2-2019-language-models]。

## 逐子层式子对比

记 $x_l$ 为第 $l$ 个块的输入，$F$ 为子层（注意力或 FFN），$\mathrm{LN}$ 沿特征维归一化。

$$\begin{aligned}
\text{post-LN：}\quad x_{l+1} &= \mathrm{LN}\bigl(x_l + F(x_l)\bigr) && \text{// 归一化作用在残差和上，恒等通路每层被重写一次}\\
\text{pre-LN：}\quad x_{l+1} &= x_l + F\bigl(\mathrm{LN}(x_l)\bigr) && \text{// 归一化只作用于分支输入，} x_l \text{ 原样进入加法}\\
\text{输出端：}\quad z &= \mathrm{LN}(x_L) && \text{// pre-LN 的最后一层归一化移到全部块之后，再进输出头}
\end{aligned}$$

差异集中在两处：归一化对象从「残差和的整体统计量」变为「分支输入的统计量」；主干 $x_l$ 在 post-LN 里必须穿过 LN，在 pre-LN 里只参与加法[^src-gpt2-2019-language-models]。

## 数据流与维度

**维度符号**：$N$ 为上下文长度（GPT-2 取 1024）；$d$ 为模型宽度（GPT-2 取 1600）；$h$ 为注意力头数，$d_h=d/h$；$d_{ff}$ 为 FFN 隐层宽度；$V$ 为词表大小（GPT-2 取 50,257）；$L$ 为块数（GPT-2 取 48）。块输入 $x_l\in\mathbb{R}^{N\times d}$，元素记 $x_{l,t,j}$，$t$ 为位置、$j$ 为通道。报告未给出多头头数与 FFN 宽度，块内按单头书写，多头展开为本页推断[^src-gpt2-2019-language-models][INFERENCE]。

### Stage 1 · 子层输入端归一化

$$\begin{aligned}
\mu_{l,t} &= \frac{1}{d}\sum_{j=1}^{d} x_{l,t,j} && \text{// } [N,d] \xrightarrow{\text{沿 } d \text{ 维逐 token 求均值}} [N,1]\ \text{——每个 token 各自一组统计量，跨 } N \text{ 不共享}\\
\sigma_{l,t} &= \sqrt{\frac{1}{d}\sum_{j=1}^{d}\bigl(x_{l,t,j}-\mu_{l,t}\bigr)^{2}} && \text{// } [N,d] \xrightarrow{\text{沿 } d \text{ 维求标准差}} [N,1]\ \text{——同一 token 内 } d \text{ 个通道的离散程度}\\
\bar{x}_l &= \gamma\odot\frac{x_l-\mu_l}{\sigma_l}+\beta && \text{// } [N,d]\to[N,d]\ \text{——} \mu_l,\sigma_l \text{ 是元素为 } \mu_{l,t},\sigma_{l,t} \text{ 的 } [N,1] \text{ 向量，} \gamma,\beta\in\mathbb{R}^{d} \text{ 沿 } N \text{ 维共享}
\end{aligned}$$

### Stage 2 · 子层只读归一化后的输入

**支路 A：多头自注意力**

$$\begin{aligned}
Q,K,V' &= \bar{x}_l W_Q,\ \bar{x}_l W_K,\ \bar{x}_l W_V && \text{// } [N,\bcancel{d}]\times[\bcancel{d},d_h]\to[N,d_h]\ \text{——消去 } d\text{，投影吃 } \bar{x}_l\text{，块输入 } x_l \text{ 本身不进子层}\\
S &= \frac{QK^{\top}}{\sqrt{d_h}}+M && \text{// } [N,\bcancel{d_h}]\times[\bcancel{d_h},N]\to[N,N]\ \text{——消去 } d_h\text{，第 } (t,s) \text{ 项是位置 } t \text{ 对位置 } s \text{ 的打分}\\
A &= \operatorname{softmax}_{s}(S) && \text{// } [N,N] \xrightarrow{\text{沿 } s \text{ 维逐行归一}} [N,N]\ \text{——因果掩码 } M \text{ 把 } s>t \text{ 的项置 } -\infty\\
\operatorname{Attn}(\bar{x}_l) &= \operatorname{Concat}_{h}\!\bigl(A_h V'_h\bigr)W_O && \text{// } [N,\bcancel{N}]\times[\bcancel{N},d_h]\to[N,d_h] \xrightarrow{\text{沿 head 维拼接}} [N,d] \xrightarrow{\times W_O\in\mathbb{R}^{\bcancel{d}\times d}} [N,d]\ \text{——消去 } N\text{，按权重混合已见位置的 value}
\end{aligned}$$

**支路 B：FFN**（块内第二个子层，输入为注意力残差和 $u_l$（Stage 3 第一行），再经 Stage 1 归一化）

$$\begin{aligned}
\operatorname{FFN}(\bar{u}_l) &= W_2\,\operatorname{GELU}(W_1\bar{u}_l) && \text{// } [N,\bcancel{d}]\times[\bcancel{d},d_{ff}]\to[N,d_{ff}]\to[N,\bcancel{d_{ff}}]\times[\bcancel{d_{ff}},d]\to[N,d]\ \text{——逐位置独立的通道混合，不跨 token}
\end{aligned}$$

### Stage 3 · 残差回加，主干逐层只做加法

$$\begin{aligned}
u_l &= x_l + \operatorname{Attn}(\bar{x}_l) && \text{// } [N,d]+[N,d]\to[N,d]\ \text{——加法两侧同形，} x_l \text{ 到 } u_l \text{ 是一条无变换的恒等通路}\\
x_{l+1} &= u_l + \operatorname{FFN}(\bar{u}_l) && \text{// } [N,d]+[N,d]\to[N,d]\ \text{——一个块内两次残差相加，宽度不变}\\
x_L &= x_0 + \sum_{l=0}^{L-1}\Bigl[\operatorname{Attn}(\bar{x}_l)+\operatorname{FFN}(\bar{u}_l)\Bigr] && \text{// } [N,d]\ \text{——主干等于 } x_0 \text{ 与 } 2L \text{ 个子层输出之和，量级随深度线性累积而非逐层重标定}
\end{aligned}$$

### Stage 4 · 输出端归一化与反嵌入

$$\begin{aligned}
z &= \mathrm{LN}(x_L) && \text{// } [N,d]\to[N,d]\ \text{——全部块之后唯一的归一化，把无归一化累加的主干拉回可读尺度}\\
\operatorname{logits} &= z\,W_E^{\top} && \text{// } [N,\bcancel{d}]\times[\bcancel{d},V]\to[N,V]\ \text{——消去 } d\text{，每行是下一 token 在 } 50257 \text{ 个词上的打分；报告未说明 } W_E \text{ 是否与输入嵌入共享}
\end{aligned}$$

## 机制理由：报告写了什么、后续分析补了什么

报告给出两条理由。其一，排布与 pre-activation 残差网络相似（He et al., 2016），归一化置于分支输入端、残差相加之前（§2.3）[^src-gpt2-2019-language-models]。其二，残差路径数值随深度累积，残差层权重初始化按 $1/\sqrt{N_{\text{res}}}$ 缩放（$N_{\text{res}}$ 为残差层数，区别于上下文长度 $N$；GPT-2 有 48 个块，报告未说明按块还是按子层计；§2.3）[^src-gpt2-2019-language-models]。两条均为排布与初始化层面的说明，报告未给出梯度或方差分析，也未做 pre-LN 与 post-LN 的对照实验（§2.3、§3 均无该消融）[^src-gpt2-2019-language-models]。

「残差主干保持恒等路径」在 pre-LN 中是代数事实：$x_l$ 与 $x_{l+1}$ 之间只隔一次加法，Stage 3 第三行写为累加形式。报告采用了该形态，但未以此论证训练收益[^src-gpt2-2019-language-models]。

「梯度范数不随深度放大」与「post-LN 需要 learning-rate warmup」是后续专门分析（arXiv:2002.04745）的结论，并非 GPT-2 报告的论断。

## 证据

GPT-2 的对应设置为：最大模型 1.5B 参数、48 层、$d_{model}=1600$，上下文 1024，残差层权重初始化乘 $1/\sqrt{N_{\text{res}}}$（残差层数），词表 50,257，batch size 512（§2.3、Table 2）[^src-gpt2-2019-language-models]。在 WebText 预训练后的零样本评测中，8 个语言建模数据集有 7 个刷新最好成绩——WikiText-2 困惑度 18.34 对旧最好 39.14、PTB 35.76 对 46.54、LAMBADA 8.63 对 99.8、enwik8 0.93 BPB、text8 0.98 BPC、WikiText-103 17.48 对 18.3；One Billion Word Benchmark 42.16 对 21.8 仍落后（Table 3、§3.1–§3.3）[^src-gpt2-2019-language-models]。这些数字对应整个模型与语料组合，各项改动未单独测量。

## 边界

报告未将 pre-LN 作为新颖点，亦未声明其在同等规模下优于 post-LN，而是类比 pre-activation 残差网络，将其作为沿用 GPT 时的改动（§2.3）[^src-gpt2-2019-language-models]。报告将零样本多任务能力定位为雏形：摘要任务定量指标仍 rudimentary，实际可用性 far from use-able（§6）[^src-gpt2-2019-language-models]。

式子对比与分界取自报告描述；报告未测量零样本表现与该排布的关系。

## 相关页面

- [[source-gpt2-2019-language-models]] — GPT-2 报告（2019，非同行评议），pre-LN 排布在本库的落点
- [[source-sundial]] — Sundial 论文源页，架构特性第 2 条列出 Pre-LN
- [[sundial]] — 时序基础模型，架构沿用 pre-LN 并报告了 pre-LN 与 post-LN 的对照 (Fig 9b)
- [[zeus]] — 时序基础模型的 Transformer block 采用 pre-LN
- [[normalization-independence]] — 时序域的另一类归一化设计（实例归一化与分布漂移）

[^src-gpt2-2019-language-models]: [[source-gpt2-2019-language-models]] — Radford et al., Language Models are Unsupervised Multitask Learners, OpenAI 技术报告, 2019
[^src-radford-2018-gpt]: [[source-radford-2018-gpt]]
