---
title: "相对位置表示（Relative Position Representation）"
type: technique
tags:
  - transformer
  - attention
  - position-encoding
  - relative-position
  - relation-aware-attention
created: 2026-10-26
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# 相对位置表示（Relative Position Representation）

Shaw、Uszkoreit 与 Vaswani（2018）把自注意力中两元素间的相对偏移表示为带标签的边，用可学习向量 $w^K_{\operatorname{clip}(j-i,k)}$、$w^V_{\operatorname{clip}(j-i,k)}$ 分别注入 key 与 value，从而完全去掉加在输入上的绝对位置编码[^src-shaw-2018-relative-position]。这也是更一般的 relation-aware self-attention（把任意两元素间的边向量注入 key/value，见 [[relation-aware-self-attention]]）在序列上的特化，边标签由裁剪后的相对偏移决定。

## 问题

Transformer 自注意力本身对输入顺序不变：把 $e_{ij}$ 写成 $q_i^\top k_j/\sqrt{d_z}$ 时，置换输入元素只置换打分矩阵的行列，输出随元素一起置换。原结构中的位置信息只通过加在输入向量上的绝对位置编码引入[^src-shaw-2018-relative-position]。绝对编码带来两个后果：

一是打分 $e_{ij}$ 同时依赖 $i$ 与 $j$ 的绝对位置，表示不随序列总长保持不变；二是位置以「第几个元素」的形式加进输入表示，序列变长后同一成分拿到的编码整体平移。

将位置表述为「两元素间的距离」可消除对总长度的依赖：$e_{ij}$ 的位置分量只由 $j-i$ 决定[^src-shaw-2018-relative-position]。

## 边标签与裁剪

**维度符号**：$n$ 序列长度；$d_x$ 输入/输出维；$d_z$ 单头输出维；$d_a$ 边向量维（论文取 $d_a=d_z$）；$h$ 注意力头数；$b$ 批大小；$k$ 裁剪距离（最大绝对相对偏移）。形状记作 $[\text{行},\text{列}]$。

$$\begin{aligned}
a^K_{ij} &= w^K_{\operatorname{clip}(j-i,\,k)}, \qquad a^V_{ij} = w^V_{\operatorname{clip}(j-i,\,k)}\\
\operatorname{clip}(x,k) &= \max\!\big(-k,\ \min(k,\,x)\big)
\end{aligned}$$

两张表 $\mathbf{w}^K=(w^K_{-k},\dots,w^K_{k})$、$\mathbf{w}^V=(w^V_{-k},\dots,w^V_{k})$ 各含 $2k+1$ 个 $d_a$ 维向量，$w^K_i,w^V_i\in\mathbb{R}^{d_a}$，不同于 Stage 1 的投影矩阵 $W^K$[^src-shaw-2018-relative-position]。索引直接使用带符号偏移而不取绝对值：$j-i>0$（key 在 query 之后，后沿）落在 $w_{1..k}$ 一侧，$j-i<0$（前沿）落在 $w_{-k..-1}$ 一侧，超过 $k$ 饱和至 $w_{\pm k}$。$2k+1$ 个标签对应「$k$ 个后沿 + $k$ 个前沿 + 1 个零偏移」，方向直接保留在表中。论文 Figure 1 以 $a_{2,1}=w_{-1}$、$a_{2,4}=w_{2}$、$a_{4,n}=w_{k}$ 为例，示意图前提为 $2\le k\le n-4$[^src-shaw-2018-relative-position]。

> 裁剪的动机是「精确的相对位置信息在超过一定距离后不再有用」，同时把 $n^2$ 条边压成 $2k+1$ 个可查表的标签[^src-shaw-2018-relative-position]。

## 数据流与维度变换

按单头书写，多头在批维展开。边向量加在 key 与 value 的**表示**上，而非打分或输入向量上。

### Stage 1 · 三组投影

$$\begin{aligned}
Q &= X W^Q && \text{// } [n, \bcancel{d_x}]\times[\bcancel{d_x}, d_z]\to[n, d_z]\ \text{——消去 } d_x\text{，每行一个 query}\\
K &= X W^K && \text{// } [n, \bcancel{d_x}]\times[\bcancel{d_x}, d_z]\to[n, d_z]\ \text{——key 侧此后还要叠边向量}\\
V &= X W^V && \text{// } [n, \bcancel{d_x}]\times[\bcancel{d_x}, d_z]\to[n, d_z]\ \text{——value 侧同样叠边向量}
\end{aligned}$$

$W^Q,W^K,W^V\in\mathbb{R}^{d_x\times d_z}$ 按层与头各一套；输入不再加正弦绝对位置编码[^src-shaw-2018-relative-position]。

### Stage 2 · 每个 head 构造内容项与位置项并相加

gather 位置 $i$ 关于全部 $j$ 的偏移向量，得到该 query 的边矩阵，再与 key 相加：

$$\begin{aligned}
\mathcal{J}_i &= \big[\operatorname{clip}(j-i,\,k)\big]_{j=1}^{n} && \text{// } [n] \xrightarrow{\text{对每个 } i \text{ 平移并裁剪}} [n]\ \text{——取值 } \{-k,\dots,k\}\text{，只有 } 2k+1 \text{ 种}\\
G^K_i &= \mathbf{w}^K[\mathcal{J}_i] && \text{// } [2k{+}1, d_a] \xrightarrow{\text{标签 } \ell \text{ 存第 } \ell{+}k \text{ 行，按 } \mathcal{J}_i{+}k\in\{0,\dots,2k\} \text{ 取 } n \text{ 行}} [n, d_a]\ \text{——第 } i \text{ 行 query 专属的边矩阵}\\
C &= QK^{\top}/\sqrt{d_z} && \text{// } [n, \bcancel{d_z}]\times[\bcancel{d_z}, n]\to[n, n]\ \text{——内容项，与标准自注意力同一次 GEMM}\\
P^{(i)} &= q_i (G^K_i)^{\top}/\sqrt{d_z} && \text{// } [bh, \bcancel{d_z}]\times[\bcancel{d_z}, n]\to[bh, n]\ \text{——位置项，每个位置一次 GEMM，} bh \text{ 合并算}\\
e_{ij} &= C_{ij} + P^{(i)}_{j} && \text{// } [n,n]\oplus[n,n]\to[n,n]\ \text{——式 (5)：两项共用同一个 } 1/\sqrt{d_z}
\end{aligned}$$

位置项无法与内容项合成单次矩阵乘：边向量随 $(i,j)$ 对变化，而单次 GEMM 要求所有 query 共享同一张 key 矩阵[^src-shaw-2018-relative-position]。论文实现将其拆为内容项一次 GEMM 加 $n$ 组 $bh\times d_z$ 与 $d_z\times n$ 的并行乘法，再用 reshape 相加[^src-shaw-2018-relative-position]。

### Stage 3 · 逐行 softmax

$$\begin{aligned}
\alpha_{ij} &= \frac{\exp e_{ij}}{\sum_{l=1}^{n}\exp e_{il}} && \text{// } [n, n] \xrightarrow{\text{沿 } j \text{ 维逐行 softmax}} [n, n]\ \text{——每行是 query } i \text{ 对全部 key 的权重}
\end{aligned}$$

打分侧位置项在 softmax 前生效，权重仍沿 key 侧归一化；value 侧 $a^V$ 不参与归一化，只改变被加权量。

### Stage 4 · value 侧加边向量、拼接、输出投影

$$\begin{aligned}
z_i &= \sum_{j}\alpha_{ij}\big(v_j + a^V_{ij}\big) && \text{// } [1, \bcancel{n}]\times[\bcancel{n}, d_z]\to[1, d_z]\ \text{——式 (3)：边向量先加到 value 再被同一组权重加权}\\
Z^{(t)} &= \big[z_i\big]_{i=1}^{n} && \text{// } n\times[d_z]\to[n, d_z]\ \text{——第 } t \text{ 个头逐行堆叠}\\
O &= [Z^{(1)},\dots,Z^{(h)}]\,W^{O} && \text{// } [n, \bcancel{h d_z}]\times[\bcancel{h d_z}, d_x]\to[n, d_x]\ \text{——多头拼接后投影回 } d_x
\end{aligned}$$

式 (3) 的边向量与 value 走同一条加权通路，因而只影响「输出里掺入多少相对位置信息」，不改变注意力权重。

## 关键推导

### 为何 $e_{ij}$ 只依赖 $j-i$

$$e_{ij}=\frac{1}{\sqrt{d_z}}\Big(\underbrace{x_i W^Q (x_j W^K)^{\top}}_{\text{内容项 } c_{ij}}+\underbrace{x_i W^Q (a^K_{ij})^{\top}}_{\text{位置项 } p_{ij}}\Big)$$

位置项中的 $a^K_{ij}$ 按定义为单变量函数 $w^K_{\operatorname{clip}(j-i,k)}$，因此 $p_{ij}$ 对位置的依赖仅剩偏移 $j-i$[^src-shaw-2018-relative-position]。边标签定义不含 $i$、$j$ 的绝对值，使位置分量与内容解耦。内容项 $c_{ij}$ 仍分别依赖 $x_i$、$x_j$ 的内容，因此 $e_{ij}$ 整体不仅是距离的函数。

在 $e_{ij}=q_i (k_j + a^K_{ij})^{\top}/\sqrt{d_z}$ 中，相对位置作为加在 key 上的偏移，与要素身份的耦合留在 $q_i$ 里。绝对正弦编码则加在 $x$ 上，经 $W^Q$、$W^K$ 后 $e_{ij}$ 的两个分量均随各自绝对位置变化。

### 裁剪后每个 query 只看 $2k+1$ 个相对位置

$\operatorname{clip}(\cdot,k)$ 的值域是 $\{-k,\dots,k\}$，共 $2k+1$ 个整数。对固定的 $i$，$\mathcal{J}_i$ 的 $n$ 个分量全部落在这 $2k+1$ 个取值内，故 $G^K_i$ 的 $n$ 行仅来自 $2k+1$ 个不同向量，且偏离超过 $k$ 的所有 key 共享同一行。参数量因此为 $2(2k+1)d_a$ 每层（base 为每头一份），与 $n$ 无关。

代价落在序列两端：位置 $i$ 的行中，$j-i$ 的取值范围是 $1-i$ 到 $n-i$，两侧被裁平导致边索引分布在边缘与居中位置不对称，两端 query 看到的「远距离」标签全部折叠到 $w_{\pm k}$。$k=0$ 时只剩 $w_0$ 一个标签，位置项退化为与 $j$ 无关的常数。

### 注意力权重分解为「内容项 × 位置项」

softmax 对加分数的恒等分解：

$$\begin{aligned}
\alpha_{ij} &= \frac{\exp(c_{ij}+p_{ij})}{\sum_{l}\exp(c_{il}+p_{il})} = \frac{\exp c_{ij}\cdot e^{\,p_{ij}}}{\sum_{l}\exp c_{il}\cdot e^{\,p_{il}}} && \text{// } [n,n]\ \text{——分母仍沿 } j \text{ 求和}\\
\phantom{\alpha_{ij}} &= \frac{1}{Z_i}\,\alpha^{\mathrm{content}}_{ij}\,e^{\,p_{ij}},\quad Z_i=\sum_{l}\alpha^{\mathrm{content}}_{il}e^{\,p_{il}} && \text{// } \alpha^{\mathrm{content}}_{ij}=\operatorname{softmax}_j(c_{ij})\ \text{——内容权重乘位置因子后重新归一化}
\end{aligned}$$

权重等于内容侧 softmax 权重 $\alpha^{\mathrm{content}}_{ij}$ 乘位置因子 $e^{p_{ij}}$ 后重新归一化。该分解是代数恒等式，不依赖任何假设。「位置项把权重推向近邻」或「注意力熵下降」要求 $p_{ij}$ 随 $j$ 系统变化；但 $p_{ij}=q_i(a^K_{ij})^{\top}/\sqrt{d_z}$ 依赖 query 内容 $x_i$，学出的 $w^K$ 方向与单调走势不受约束，论文也未报告注意力权重分布或熵的测量[^src-shaw-2018-relative-position]，未提供「局部性偏置」证据。

这个分解还暴露了它与标量位置偏置的结构差别：

$p_{ij}$ 依赖 $q_i$，不是所有 query 共享的距离曲线，故不能写成 $\alpha_{ij}\propto\exp(c_{ij})\,g(j-i)$ 的单一 $g$。$a^K_{ij}$ 里不含 $x_j$，无法吸收进作用于 key 的位置相关线性映射 $W(j-i)k_j$；在 [[generalized-positional-encoding-framework|GPE 框架]] 的 $A(n)=f(n)\,q_i^\top W(n)k_j+b(n)$ 形式里，它既不是纯加性标量 $b(n)$，也不在 $W(n)$ 的作用范围内[INFERENCE]。与 [[alibi|ALiBi]] 的 $m\cdot(j-i)$ 相比，边向量随头学习、维数为 $d_z$，且同时进入 value。

## 参数与代价

边向量取 $d_a=d_z$ 与 key/value 同宽，加法无需额外投影，不需额外线性变换[^src-shaw-2018-relative-position]。base 模型每层每头一套独立边表示，big 模型每层一套并由头间共享，边表示还可跨序列共享[^src-shaw-2018-relative-position]。按头共享后存储从 $O(hn^2d_a)$ 降到 $O(n^2d_a)$，自注意力整体空间从 $O(bhnd_z)$ 增到 $O(bhnd_z+n^2d_a)$，$d_a=d_z$ 时相对增幅由 $n/(bh)$ 决定[^src-shaw-2018-relative-position]。步速下降约 7%，靠保持模型与批大小不变在 P100 上完成训练[^src-shaw-2018-relative-position]。

## 证据

clipping 距离（base 模型，EN-DE 开发集 newstest2013）：

| $k$ | 0 | 1 | 2 | 4 | 16 | 64 | 256 |
|---|---|---|---|---|---|---|---|
| BLEU | 12.5 | 25.5 | 25.8 | 25.9 | 25.8 | 25.9 | 25.8 |

$k\ge 2$ 后基本不变，$k=0$ 降至 12.5[^src-shaw-2018-relative-position]。$k=0$ 时位置项为行内常数，softmax 对该常数平移不变，等价于纯内容注意力；由于模型已去掉绝对位置编码，此时完全无位置信息，与表 3 中两个表示都不用的 12.5 相互印证[INFERENCE]。

两条注入路径的消融（base 模型，EN-DE 开发集）：

| $a^V$ | $a^K$ | BLEU |
|---|---|---|
| 是 | 是 | 25.8 |
| 否 | 是 | 25.8 |
| 是 | 否 | 25.3 |
| 否 | 否 | 12.5 |

仅在 compatibility 注入 $a^K$ 即可获得全部增益，去掉 $a^K$ 下降 0.5，仅保留 $a^V$ 而不保留 $a^K$ 同样无效；论文据此提出或可仅在打分侧注入，仍待其他任务进一步验证[^src-shaw-2018-relative-position]。

## 适用范围

实验仅在 WMT 2014 EN-DE 与 EN-FR 两个翻译任务上验证；$k$ 敏感度及双通路消融均基于开发集 newstest2013，主表采用测试集 newstest2014[^src-shaw-2018-relative-position]。「裁剪同时让模型能泛化到训练未见过的序列长度」仅为动机陈述，论文未提供外推实验[^src-shaw-2018-relative-position]。叠加绝对位置编码「未观察到进一步收益」缺乏对应表格，Table 1 仅对比了纯绝对与纯相对配置[^src-shaw-2018-relative-position]。其他模态、图结构或长上下文外推均未覆盖，[[relation-aware-self-attention|relation-aware self-attention]] 的图推广属论文列出的未来工作。

## 相关页面

[[source-shaw-2018-relative-position]] 为论文摘要；[[relation-aware-self-attention]] 是把边信息注入自注意力的一般形式；[[alibi]] 把相对偏置简化为不学习的标量斜率；[[rope]] 用旋转把位置放进内积而非加到表示上；[[generalized-positional-encoding-framework]] 对应位置编码的乘性/加性分解；[[scaling-factor-sqrt-dk]] 包含内容项与位置项共用的 $1/\sqrt{d_z}$；[[transformer-architecture]] 是绝对位置编码的原始设定；[[sinusoidal-positional-encoding]] 为被本方法替换的绝对位置编码；[[context-window-extension]] 则是相对位置编码用于长度外推的后续脉络。

[^src-shaw-2018-relative-position]: [[source-shaw-2018-relative-position]]（arXiv:1803.02155v1，PDF 归档 `raw/shaw-2018-relative-position-1803-02155.pdf`）
