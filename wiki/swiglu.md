---
title: "SwiGLU（GLU 变体前馈层）"
type: technique
tags:
  - feed-forward
  - activation-function
  - gated-linear-unit
  - transformer
  - architecture
created: 2026-10-26
last_updated: 2026-10-26
source_count: 2
confidence: medium
status: active
---

# SwiGLU 与 GLU 变体前馈层

SwiGLU 是 [[glu-gated-linear-unit|GLU]] 将门控分支的 sigmoid 替换为 $\operatorname{Swish}_1(x)=x\sigma(x)$ 的变体，用作 Transformer 的逐位置 FFN：两条线性投影升至内层宽度，其中一条先经激活，逐元素相乘后由第三个矩阵降回残差流宽度（arXiv:2002.05202，式 5–6）[^src-shazeer-2020-glu-variants]。同族的 ReGLU、GEGLU、Bilinear 分别将门控激活换为 ReLU、GELU 与恒等，五个变体的差异仅在 $\phi$ 处（式 4–6）[^src-shazeer-2020-glu-variants]。

## 数据流与维度

**维度符号**：$N$ 序列长度（token 数）；$d_{\text{model}}$ 残差流宽度；$d_{ff}$ 内层宽度。全页沿用行向量约定：每个 token 占一行，权重右乘，形状记作 $[\text{行},\text{列}]$。$\phi$ 为门控激活，取 $\sigma$、$\mathrm{ReLU}$、$\mathrm{GELU}$、$\operatorname{Swish}_1$ 或恒等之一[^src-shazeer-2020-glu-variants]。

### Stage 1 · 两条升维投影，门控支路先过激活

**支路 A：门控支路**

$$\begin{aligned}
G &= \phi\!\left(HW\right) && \text{// } [N, \bcancel{d_{\text{model}}}]\times[\bcancel{d_{\text{model}}}, d_{ff}]\to[N, d_{ff}]\ \text{——每个 token 得到 } d_{ff} \text{ 个门控系数}
\end{aligned}$$

**支路 B：内容支路**，读同一个 $H$、同样的权重形状，只差不过激活：

$$\begin{aligned}
U &= HV && \text{// } [N, \bcancel{d_{\text{model}}}]\times[\bcancel{d_{\text{model}}}, d_{ff}]\to[N, d_{ff}]\ \text{——与 } G \text{ 同形，保留未门控的线性像}
\end{aligned}$$

两条支路各占一个 $[d_{\text{model}}, d_{ff}]$ 权重矩阵，参数上等价于将一次 $[d_{\text{model}}, 2d_{ff}]$ 投影 split 成两半：FaST 的 GLU expert 即采用此写法（$W_F\in\mathbb{R}^{d\times 2ed}$，split 后一半过 $\sigma$）[^src-fast-long-horizon-forecasting]。

### Stage 2 · 逐元素门控相乘

$$\begin{aligned}
Z &= G\odot U && \text{// } [N, d_{ff}]\odot[N, d_{ff}]\to[N, d_{ff}]\ \text{——第 } j \text{ 个通道按 } G_{:,j} \text{ 缩放 } U_{:,j}\text{，}\phi=\mathrm{id} \text{ 时退化为逐通道二次型（Bilinear）}
\end{aligned}$$

逐元素积是唯一引入两路交互的位置：ReLU/GELU/Swish 的 FFN 此处只有逐通道标量非线性，没有跨支路乘积（式 1–3）[^src-shazeer-2020-glu-variants]。

### Stage 3 · 降回残差流宽度

$$\begin{aligned}
F &= ZW_2 && \text{// } [N, \bcancel{d_{ff}}]\times[\bcancel{d_{ff}}, d_{\text{model}}]\to[N, d_{\text{model}}]\ \text{——逐位置独立，结果加回残差流}
\end{aligned}$$

三个矩阵全部逐位置共享、层间不共享。实验采用无偏置约定：定义式 4–5 保留偏置 $b$、$c$，式 6 去掉门控分支、内容分支与降维投影的偏置[^src-shazeer-2020-glu-variants]。

## 变体形式

| 层 | 门控激活 $\phi$ | 无偏置形式（式 6） |
|---|---|---|
| GLU | $\sigma$ | $(\sigma(xW)\otimes xV)W_2$ |
| Bilinear | 恒等 | $(xW\otimes xV)W_2$ |
| ReGLU | $\mathrm{ReLU}$ | $(\max(0,xW)\otimes xV)W_2$ |
| GEGLU | $\mathrm{GELU}(x)=x\Phi(x)$ | $(\mathrm{GELU}(xW)\otimes xV)W_2$ |
| SwiGLU | $\operatorname{Swish}_1(x)=x\sigma(x)$ | $(\operatorname{Swish}_1(xW)\otimes xV)W_2$ |

五个变体完全同构：两条并列的升维投影、一次逐元素积、一次降维投影，唯一差异是门控分支上的 $\phi$（式 6）。Bilinear 把 $\phi$ 取为恒等，是唯一不带门控非线性的变体，在表中分开了「门控非线性」与「两路相乘」两个因素[^src-shazeer-2020-glu-variants][INFERENCE]。偏置虽见于定义式，但所有对照实验均采用无偏置版本，变体间的比较不受偏置项干扰（式 4–6）。

## 参数与 FFN 宽度折算

两矩阵 FFN 每层参数量为 $2d_{\text{model}}d_{ff}$，三矩阵变体为 $3d_{\text{model}}d'_{ff}$；令两者相等得 $d'_{ff}=\tfrac{2}{3}d_{ff}$。T5-base 基准为 $d_{ff}=3072$，门控变体折至 $d_{ff}=2048$，参数量与乘法次数同时对齐（式 6 下方、§3.1）[^src-shazeer-2020-glu-variants]。

折算后内层总通道为 $2d'_{ff}=\tfrac{4}{3}d_{ff}$，比基准更宽；但经激活的只有 $d'_{ff}=\tfrac{2}{3}d_{ff}$ 一路，另一路 $d'_{ff}$ 是不带非线性的内容支路。等参数对照因此同时改动两个量：非线性通道数下降、参与门控乘积的通道数上升。

## 实验证据

实验设置（§3.1–3.3）：T5-base 编解码器各 12 层，$d_{\text{model}}=768$、$h=12$、$d_k=d_v=64$；C4 span-filling 预训练 524,288 步，batch 128（输入 512 token、输出 114 token），采用 Adafactor 与逆平方根学习率、末 10% 线性衰减；不同于 Raffel et al.，预训练不用 dropout[^src-shazeer-2020-glu-variants]。另为各变体训练 4 个 65,536 步模型测跨运行方差。微调 131,072 步、lr $10^{-3}$、每步约 65,536 token，层输出、FFN 隐层与注意力权重 dropout 均设 0.1，固定嵌入；SQuAD 与 GLUE、SuperGLUE 按样本比例混合为单任务，各模型微调一次，报告 dev 最佳 checkpoint 分数[^src-shazeer-2020-glu-variants]。

**表 1**（C4 heldout log-perplexity，全部模型等参数等算力）：

| FFN | 65,536 步（4 次运行均值/标准差） | 524,288 步 |
|---|---|---|
| ReLU（基线） | 1.997 (0.005) | 1.677 |
| GELU | 1.983 (0.005) | 1.679 |
| Swish | 1.994 (0.003) | 1.683 |
| GLU | 1.982 (0.006) | 1.663 |
| Bilinear | 1.960 (0.005) | 1.648 |
| GEGLU | 1.942 (0.004) | 1.633 |
| SwiGLU | 1.944 (0.010) | 1.636 |
| ReGLU | 1.953 (0.003) | 1.645 |

**表 2–4**（微调后 dev 分数）：

| FFN | GLUE 均分（表 2） | SuperGLUE 均分（表 3） | SQuAD EM/F1（表 4） |
|---|---|---|---|
| ReLU（基线） | 83.80 | 72.76 | 83.18 / 90.87 |
| GELU | 83.86 | 72.98 | 83.09 / 90.79 |
| Swish | 83.60 | 72.40 | 83.25 / 90.76 |
| GLU | 84.20 | 73.95 | 82.88 / 90.69 |
| Bilinear | 83.79 | 73.81 | 83.82 / 91.06 |
| GEGLU | 84.12 | 73.96 | 83.55 / 91.12 |
| SwiGLU | 84.36 | 74.56 | 83.42 / 91.03 |
| ReGLU | 84.67 | 73.66 | 83.53 / 91.18 |

对照行来自 [Raffel et al., 2019]：GLUE 83.28、SuperGLUE 71.36、SQuAD 80.88/88.81；表内标准差（0.235、0.416、0.343/0.226）取自其多次运行[^src-shazeer-2020-glu-variants]。

524,288 步档五个门控变体的困惑度全部低于 ReLU 基线 1.677、GELU 1.679 与 Swish 1.683，最低为 GEGLU 1.633 与 SwiGLU 1.636；65,536 步档同样五个变体全部低于三个基线，变体内部排序一致（GEGLU 1.942 < SwiGLU 1.944 < ReGLU 1.953 < Bilinear 1.960 < GLU 1.982），三个基线在该档的相对次序与长程档相反（GELU 1.983 < Swish 1.994 < ReLU 1.997）[^src-shazeer-2020-glu-variants]。微调任务中 GLUE 均分以 ReGLU 84.67、SwiGLU 84.36 最高，SuperGLUE 以 SwiGLU 74.56 最高，SQuAD EM 以 Bilinear 83.82 最高、F1 以 ReGLU 91.18 最高；GLU 在 SQuAD 两项均低于 ReLU 基线（82.88/90.69）[^src-shazeer-2020-glu-variants]。困惑度上五个变体一致领先，下游分数多数领先但存在反例（如 SQuAD 的 GLU），该结果仅覆盖 T5-base 的文本 span-filling 预训练与下游微调[^src-shazeer-2020-glu-variants]。

## 为什么门控有效

论文结论节明确不解释门控变体为何有效，相关成因未见于原论文，亦未经实验检验[^src-shazeer-2020-glu-variants][INFERENCE]。

乘性交互体现为 $G\odot U$ 让单层 FFN 出现 $x$ 的两个线性像之积，Bilinear 情形下每个内层通道即为一个秩 1 二次型，ReLU/GELU 的单层 FFN 并无此类输入间交叉项；门与内容分离表现为门控支路只按内容调幅度，内容支路保留完整线性像，等价于把容量从「更多非线性通道」挪向「调制已有通道」；梯度路径上，$Z=G\odot U$ 对 $U$ 的梯度正比于 $G$、对 $G$ 的梯度正比于 $U$，两路互为缩放因子，是训练行为可能更稳的一种来源。

## 边界

评测仅采用 T5-base 规模的文本 span-filling 预训练以及文本理解与抽取式问答微调，未报告更大模型、其他模态或非文本任务。表 1 中 524,288 步列为单次运行，方差仅由 65,536 步列的 4 次运行给出；微调每模型单次且取 dev 最佳 checkpoint，结果带有噪声，表内标准差行未覆盖各变体[^src-shazeer-2020-glu-variants]。与 [Raffel et al., 2019] 行不可直接对比：同为 FFN_ReLU，对方预训练使用 dropout 导致分数偏低[^src-shazeer-2020-glu-variants]。等参数折算只针对三个权重矩阵与 $d_{ff}$；无偏置下该折算精确，但未分别对齐非线性通道数与门控通道数。

## 相关页面

[[glu-gated-linear-unit]] 关联 GLU 定义与 Mamba 输入门关系；[[gated-linear-units]] 给出 GLU 作为 MoE expert（FaST）时的合并投影写法；[[transformer-architecture]] 对应 Stage 4 的 ReLU 前馈与本页门控前馈；[[gated-residual-network]] 包含时序模型里以 GLU 门控残差分支的构建块；[[source-shazeer-2020-glu-variants]] 为本页来源。

[^src-shazeer-2020-glu-variants]: [[source-shazeer-2020-glu-variants]]
[^src-fast-long-horizon-forecasting]: [[source-fast-long-horizon-forecasting]]
