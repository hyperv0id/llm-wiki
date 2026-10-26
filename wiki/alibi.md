---
title: "ALiBi"
type: entity
tags:
  - transformer
  - position-embedding
  - extrapolation
  - iclr-2022
created: 2026-04-28
last_updated: 2026-10-26
source_count: 5
confidence: high
status: active
---

# ALiBi (Attention with Linear Biases)

ALiBi 是 Ofir Press、Noah A. Smith、Mike Lewis 提出的位置编码方法（ICLR 2022）：在短序列上训练，推理时外推至训练长度的数倍，且同长度下困惑度不劣于 sinusoidal[^src-alibi]。

## 核心思想

ALiBi 不在词嵌入中加位置编码，而是在 query-key 打分后、softmax 前加上与两者距离成线性关系的偏置[^src-alibi]：

$$\operatorname{softmax}\left(q_i K^{\top} + m\cdot[-(i-1),\dots,-2,-1,0]\right)$$

$q_i\in\mathbb{R}^{1\times d}$ 是第 $i$ 个 query，$K\in\mathbb{R}^{i\times d}$ 是因果可见的前 $i$ 个 key，$m$ 是该头的斜率，训练前设定、不参与学习[^src-alibi]。偏置直接加在 logits 上，不乘论文式 (1) 中的 $1/\sqrt{d_k}$ 缩放[^src-alibi]；网络其余部分不变，输入端也不加位置向量[^src-alibi]。

## 数据流与维度变换

**维度符号**：$L$ 为输入子序列长度，$d$ 为单头通道维，$n$ 为注意力头数，$d_{model}=nd$；$i$ 为 query 下标、$j$ 为 key 下标，$\Delta=i-j\ge 0$ 为相对距离。形状记作 $[\text{行},\text{列}]$，对齐论文 §3 单头记号；掩码并入偏置见 §3 实现段[^src-alibi]。

### Stage 1 · 投影得 q/k/v

输入 $X$ 只有 token embedding，不含位置成分，三条投影与标准注意力完全一致[^src-alibi]：

$$\begin{aligned}
Q &= XW^{Q},\quad K = XW^{K},\quad V = XW^{V} && \text{// } [L, d_{model}]\times[d_{model}, d_{model}]\to[L, d_{model}]\ \text{——位置方法只改下游偏置，不动这三条投影}\\
q_i &= Q_{i,:}^{(h)}\in\mathbb{R}^{1\times d},\quad K^{(h)}=K_{1:L,:}^{(h)}\in\mathbb{R}^{L\times d} && \text{// } [L, d_{model}]\xrightarrow{\text{按头切片}}[L, d]\ \text{——第 } i \text{ 个 query 只与 } j\le i \text{ 的 key 打分，}j>i \text{ 由因果掩码屏蔽}
\end{aligned}$$

### Stage 2 · 造加性偏置矩阵

偏置只由相对距离决定，每个头一条斜率[^src-alibi]：

$$\begin{aligned}
D_{ij} &= -(i-j) && \text{// } [L, L]\ \text{——第 } i \text{ 行即 } [-(i-1),\dots,-1,0]\text{；}j>i \text{ 处置 } -\infty \text{ 由因果掩码给出}\\
B^{(h)} &= m_h\,D && \text{// } [L, L]\ \text{——同一个矩阵按头缩放，不含绝对位置，因此没有为 } j>L \text{ 预留的表项}\\
B &= \operatorname{stack}_{h=1}^{n}B^{(h)} && \text{// } n\times[L,L]\to[n,L,L]\ \text{——等价于 } [n,1,1]\times[1,L,L] \text{ 沿头维广播；实现里并进掩码，掩码由 } L\times L \text{ 变 } n\times L\times L
\end{aligned}$$

### Stage 3 · 加进 logits 再按 key 维 softmax

$$\begin{aligned}
S^{(h)} &= \frac{Q^{(h)}K^{(h)\top}}{\sqrt{d}} + B^{(h)} && \text{// } [L,\bcancel{d}]\times[\bcancel{d},L]\to[L,L]\ \text{——消去通道维得打分矩阵，偏置与打分同处 logits 空间直接相加}\\
A^{(h)} &= \operatorname{softmax}\!\left(S^{(h)}\right) && \text{// } [L,L]\xrightarrow{\text{沿 key 维 } j \text{ 逐行归一化}}[L,L]\ \text{——第 } i \text{ 行权重和为 1，偏置改的是分配而非总和}
\end{aligned}$$

### Stage 4 · 加权求和并合并头

$$\begin{aligned}
O^{(h)} &= A^{(h)}V^{(h)} && \text{// } [L,\bcancel{L}]\times[\bcancel{L},d]\to[L,d]\ \text{——消去 key 维；}V\text{ 不带位置信息，位置只经由 } A \text{ 进入输出}\\
O &= \operatorname{concat}_{h}\!\left(O^{(h)}\right)W^{O} && \text{// } [L, nd]\times[nd, d_{model}]\to[L, d_{model}]\ \text{——多头拼接回模型维}
\end{aligned}$$

与标准 transformer 的差别只在 $B$：偏置加在 logits 上，不改 $K$ 投影也不进 $V$。前向只多出一次 $[n,L,L]$ 加法且无新增算子，同 $L$ 下训练速度与显存基本不变（见下「开销」），代价是多出 $n\times L\times L$ 掩码[^src-alibi]。

## 几何斜率序列

$n$ 个头的斜率构成几何序列，起点为 $2^{-8/n}$，公比也取 $2^{-8/n}$[^src-alibi]：

$$m_h = 2^{-8h/n},\quad h=1,\dots,n \quad\Longrightarrow\quad m_1 = 2^{-8/n},\;\; m_n = 2^{-8}$$

8 头为 $2^{-1},2^{-2},\dots,2^{-8}$（公比 $1/2$）；16 头取 8 头序列相邻两项的几何平均得 $2^{-0.5},2^{-1},\dots,2^{-8}$，起点与公比同为 $2^{-0.5}$。

公比等于起点，且 $h$ 走到 $n$ 时指数恰好到 $-8$，序列右端对所有 $n$ 都钉在 $2^{-8}$：最慢衰减的头不随头数变化，增加头数是在短距离端加密。论文将该固定序列与 sinusoidal 的波长几何级数并列——后者同样设定一次后跨模型规模沿用[^src-alibi]。

### 斜率决定距离尺度

偏置以 $m_h\Delta$ 直接进入 logits，距离 $\Delta$ 的 key 相较等打分、距离 0 的 key 被压低 $e^{-m_h\Delta}$ 倍。把"惩罚满 1 nat"作为该头的距离尺度，得

$$\Delta_h^{\ast} = \frac{1}{m_h} = 2^{8h/n}$$

8 头的尺度是 $2,4,\dots,256$ 个 token，16 头是 $1.41,2,\dots,256$。该尺度仅反映偏置单独造成的倍数关系；真实权重还取决于数据相关的 $q_ik_j$，整体尺度由训练决定，论文未给出将 nats 校准到具体距离的推导[^src-alibi]。

### 为什么取几何级、为什么落在 $(0,1)$

论文未为这套斜率给出定理，只交代来源：先设斜率可训练，外推失败且训练慢 3%；手工探索约 10 组斜率后发现，有效斜率都落在 $(0,1)$ 且越靠近 0 越密；按指数分布随机采样在部分设置下可用，但方差大[^src-alibi]。公比等于起点的几何级数正好在 $\log m$ 上给出等距衰减率网格，自然满足越靠近 0 越密。这套斜率在 WikiText-103 上选定后，原样用于书籍语料与 1.3B 模型，未再调参[^src-alibi]。

## 外推能力从哪来，到哪里为止

论文给出三条机制[^src-alibi]：

1. 偏置只依赖相对距离 $\Delta$，不含绝对位置，因此不存在"位置超过 $L$ 就没有对应表项"的区间；learned 与 sinusoidal 都在输入端按绝对位置给向量，$j>L$ 落在训练分布之外。
2. 位置信息每层注入 Q、K，不进 V；输出是 value 的加权和，本身不带显式位置。
3. 惩罚随距离无界增长，形成对近邻的偏好（recency bias），不同头以不同速率加压。

[INFERENCE] 三条中仅第 1 条直接对应“外推”。rotary 与 T5 bias 同样满足第 2 条，实测分别仅多延伸 $k=200/100$ 与 $k=600/800$ 个 token（$L=512/1024$，§2.2），“相对位置 + 每层注入”本身无法实现数倍于训练长度的外推[^src-alibi]。

Appendix B 的滑窗实验划出边界：改用 stride $S=1$ 的滑窗让每个预测都获得最长可用上下文后，ALiBi 的困惑度随 $L_{valid}$ 基本持平——$L=512$ 从 17.98（$L_{valid}=512$）到 18.30（3072），$L=1024$ 从 17.46 到 17.92，$L=3072$ 为 16.96（Table 15）；同一设置下 sinusoidal 仍从 18.35 涨到 360.12（Table 13）[^src-alibi]。非重叠评估下的收益主要来自缓解 early token curse：更长子序列降低了“开局缺上下文”的预测占比。据此，$L_{valid}>L$ 时 ALiBi 未必真的用上了超过训练长度的上下文[^src-alibi]。

## 与 sinusoidal / rotary / T5 bias 的实测对照

WikiText-103 验证集，$L=3072$ 训练、非重叠评估（Table 4）[^src-alibi]：

| $L_{valid}$ | Sinusoidal | Rotary | T5 Bias | ALiBi |
|---|---|---|---|---|
| 3072 | 18.67 | 18.57 | 18.01 | 17.60 |
| 4072 | 28.59 | 18.59 | 17.88 | 17.48 |
| 8072 | 102.74 | 21.60 | 18.06 | 17.30 |
| 16072 | 231.29 | 34.51 | OOM | 17.22 |

三种基线能外推的距离按 §2.2 顺序排列：sinusoidal 仅外推 $k\approx 20$（$L=512$）/ $k\approx 50$（$L=1024$）便开始恶化，rotary 为 $k\approx 200/100$，T5 bias 为 $k\approx 600/800$ 但训练慢一倍以上（$L=3072$ 时 4.3k 对 15.3k WPS，Table 1）[^src-alibi]。上表这组 $L=3072$ 中，sinusoidal 到 4072 已升至 28.59，rotary 到 8072 升至 21.60；ALiBi 从 17.60 降至 17.15（13072），到 16072 为 17.22，未出现退化。训练长度更短的模型也追平更长基线：$L=512$ 的 ALiBi 在 $L_{valid}=3072$ 上为 18.40，低于 sinusoidal $L=3072$ 的 18.67，且训练快 1.84 倍（§4.1、Table 5）[^src-alibi]。

## 特性

| 特性 | 描述 |
|------|------|
| **归纳偏置** | 对近邻位置有偏好（recency）[^src-alibi] |
| **无位置嵌入** | 网络任何位置都不加位置向量[^src-alibi] |
| **相对位置** | 每层注入位置信息到 Q 和 K，不注入 V[^src-alibi] |
| **固定斜率** | 斜率训练前设定，不参与学习[^src-alibi] |
| **偏置缩放** | 偏置不乘 $1/\sqrt{d_k}$[^src-alibi] |

## 时序预测中的扩展

ALiBi 的斜率与"加性偏置"形式用于时间序列预测，改动集中在距离项：

| 做法 | 距离项 | 出处 |
|---|---|---|
| 原式 | $m_k\cdot\lvert i-j\rvert$ | [[linear-attention-bias\|线性注意力偏置]] |
| 周期折叠 | $m_k\cdot\min\!\left(\lvert i-j\rvert \bmod P_S,\ P_S-\lvert i-j\rvert\right)$ | [[periodic-nested-attention-bias\|周期化相对注意力偏置]]（PENGUIN，AISTATS 2026）[^src-penguin] |
| 局部性先验 | 距离衰减用于金融时序的局部归纳偏置 | TIPS（AAAI 2026）[^src-tips] |

PENGUIN 直接沿用斜率公式 $m_k=2^{-k/8}$ 且固定不训练，只在多头之上增加一层"组"：每组绑定一个周期长度并共享该组的 $K,V$，在单次注意力中同时处理多个周期[^src-penguin]。消融显示偏置项贡献最大（Traffic avg MSE 0.407→0.396），周期形式进一步降至 0.390[^src-penguin]。细节见 [[penguin|PENGUIN]]。

## 性能对比

### WikiText-103 非重叠评估（Table 5）

| 训练长度 $L$ | 512 | 1024 | 1536 | 2048 | 3072 |
|---|---|---|---|---|---|
| 64 | 22.09 | 21.73 | 21.63 | 21.59 | 21.53 |
| 512 | 19.73 | 18.81 | 18.50 | 18.48 | 18.40 |
| 1024 | – | 18.66 | 18.20 | 18.05 | 17.96 |
| 3072 | – | – | – | – | 17.60 |
| Sinusoidal$^{\ast}$ | 20.05 | 19.34 | 19.05 | 18.87 | 18.67 |
| Rotary$^{\ast}$ | 20.07 | 19.33 | – | – | 18.57 |
| T5 Bias$^{\ast}$ | 19.65 | 18.80 | – | – | 18.01 |

$^{\ast}$ 基线仅在 $L_{valid}=L$ 处取指标，不测外推[^src-alibi]。表中省略 $L=128/256/1536/2048$ 四行，完整 8 个模型见 Table 5。$L=64$ 训练的模型外推至 3072（48 倍）仍为 21.53，低于 $L_{valid}=64$ 时的 28.46，外推方向单调改善；$L=512$ 训练的模型在 $L_{valid}=512$ 处为 19.73，低于同 $L$ 的 sinusoidal（20.05）[^src-alibi]。

### 开销

| 指标 | Sinusoidal $L=512$ | ALiBi $L=512$ |
|---|---|---|
| 训练速度（WPS） | 28.5k | 28.3k |
| 评估速度（WPS） | 82.1k | 85.8k |
| 训练显存 | 15.3 GB | 15.3 GB |
| 额外参数 | 0 | 0 |

同 $L$ 下两者速度持平，ALiBi 仅多出 $n\times L\times L$ 掩码[^src-alibi]。在 1.3B / CC100+RoBERTa 上，收益来自更短序列训练：$L=512$ 的 ALiBi 与 $L=1024$ 的 sinusoidal 训练时长相同（5.5k 小时，Table 11），显存为 24.6 对 26.2 GB，$L_{valid}=1024$ 时验证困惑度为 9.30 对 9.24（差 0.06，§4.2）；换到 $L_{valid}=2048$ 为 9.54 对 48.85（Table 12）[^src-alibi]。$L=1024$ 的 ALiBi 对比 $L=2048$ 的 sinusoidal 为 8.92 对 9.01，显存少 3.1 GB，达到给定困惑度的平均耗时减少 11%（Table 11、§4.2）[^src-alibi]。同 $L$ 训练时两者接近：$L=2048$ 时为 8.84 对 8.83（Table 12）[^src-alibi]。

## 应用场景

长输入推理在训练短序列后直接处理更长子序列，节省显存与训练时间[^src-alibi]。免长序列预训练仅训练 $L=1024$ 即可覆盖 GPT-3 的 2048 输入长度，节省的显存与耗时源于更短子序列，而非额外技巧（§4.2、Table 11）[^src-alibi]。

## 与其他位置编码的关系

Sinusoidal 属输入端一次注入的绝对位置，仅多维持几十个 token 便转为退化[^src-alibi]；Rotary 属每层乘入 Q/K 的相对位置，多维持约 200 个 token，训练与推理更慢[^src-alibi]；T5 Bias 按距离查学习到的相对位置偏置，多维持约 600–800 个 token，训练慢一倍以上[^src-alibi]；ALiBi 属零参数的加性线性相对位置偏置，同 $L$ 无速度损失[^src-alibi]。

[[relative-position-representation|相对位置表示]]（Shaw et al. 2018）是这条加性路径的原始形式：$2k+1$ 个可学习向量按 $\operatorname{clip}(j-i,k)$ 加到 $K$ 与 $V$ 上，向量随层（base 模型还随头）学习[^src-shaw-2018-relative-position][^src-alibi]。ALiBi 把它压成每个头一个固定斜率、只改 logit、不进 value。

## 局限性

峰值位置随设置变化：CC100+RoBERTa 上 $L=512$/$1024$ 模型在约 $2L$ 处最好（1012 时 9.3、2024 时 8.9），WikiText-103 上至少到 $3L$ 仍在改善（§4.1、§4.2）[^src-alibi]。滑窗评估下困惑度随 $L_{valid}$ 持平，说明 $L_{valid}>L$ 收益主要来自 early token curse 缓解，而非更长有效上下文（Appendix B、Table 15）[^src-alibi]。实验只覆盖语言建模三个语料（WikiText-103、Toronto Books、CC100+RoBERTa），未测其他任务与架构[^src-alibi]。在 [[generalized-positional-encoding-framework|GPE 框架]] 视角下，ALiBi 无法满足 [[long-distance-correlation-preservation|LDCP]] 和 [[gradient-positional-sensitivity|GPS]]，导致远距离相关性被完全抑制且梯度不携带位置信息[^src-vetcha-2026-towards-infinite-length-extrapolation]。

## 引用

[^src-alibi]: [[source-alibi]]
[^src-penguin]: [[source-penguin]]
[^src-tips]: [[source-tips]]
[^src-vetcha-2026-towards-infinite-length-extrapolation]: [[source-vetcha-2026-towards-infinite-length-extrapolation]]
[^src-shaw-2018-relative-position]: [[source-shaw-2018-relative-position]]
