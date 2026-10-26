---
title: "Sparse Transformer 的 Strided 与 Fixed 稀疏注意力"
type: technique
tags:
  - attention
  - sparse-attention
  - efficient-transformer
  - long-sequence
  - autoregressive
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Sparse Transformer 的 Strided 与 Fixed 稀疏注意力

**Sparse Transformer**（Child et al., 2019）将 [[transformer-architecture|Transformer]] 解码器的全注意力 $S_i=\{j: j\le i\}$ 替换为 $p$ 个稀疏头的组合：第 $m$ 个头仅对依赖集 $A^{(m)}_i\subset\{j: j\le i\}$ 计算注意力。两项约束为：任意输入位置 $j$ 均能经由至多 $p+1$ 个位置的路径连到任意输出位置 $i$，且 $|A^{(m)}_i|\propto\sqrt{n}$。二者共同将 $O(n^2)$ 降至 $O(n\sqrt{n})$ 且保持全局连通（§4.2）[^src-child-2019-sparse-transformer]。论文只实现 $p=2$ 的两种分解：strided 与 fixed（Figure 3）。

## Strided 模式

$$A^{(1)}_i = \{t, t+1, \dots, i\},\qquad t = \max(0,\ i-l)$$

$$A^{(2)}_i = \{\,j : (i-j) \bmod l = 0\,\}$$

$l$ 为 stride，取值接近 $\sqrt{n}$（§4.3）。第一个头看长度 $l$ 的局部窗口，第二个头看与 $i$ 同“列”的全部历史位置，适合图像等天然按 stride 对齐的数据。对文本这类无周期结构的数据，元素空间坐标与其未来最相关位置不必对应，strided 会路由失败，因此文本实验改用 fixed（§4.3，Table 2）。

## Fixed 模式

$$A^{(1)}_i = \{\,j : \lfloor j/l \rfloor = \lfloor i/l \rfloor\,\}$$

$$A^{(2)}_i = \{\,j : j \bmod l \in \{t, t+1, \dots, l\}\,\},\qquad t = l-c$$

每块末尾的少量单元充当“摘要单元”（summary cells）：先在块内汇总，再广播给所有后续位置（§4.3）。例如 $l=128$、$c=8$ 时，所有大于 128 的位置都能看到 120–128 这 9 个位置。因 $c=1$ 表达力受限——多数表示只被单块使用、少数位置被所有块使用，论文实际取 $c\in\{8,16,32\}$、$l\in\{128,256\}$，代价是比 strided 约多 $c$ 倍计算；多头时让各头看块内长度 $c$ 的不同子块，优于都看同一子块（§4.3）。

## 连通性与每列依赖集

按 §4.2 连通性判据，对每对 $j\le i$，存在下标路径 $(j, a, b, \dots, i)$（至多 $p+1$ 个位置）使 $j\in A^{(1)}_a$、$a\in A^{(2)}_b$ 等逐段衔接，即信号在常数步内从任意位置传到任意位置。

Strided 可两步连通[INFERENCE]：按 §4.2 判据取路径 $(j, a, i)$，$a = i - l\lfloor(i-j)/l\rfloor$。在前提 $i\ge l$ 下，因 $(i-a)\bmod l=0$ 且 $a\le i$，故 $a\in A^{(2)}_i$；令 $i-j=ql+d$（$q\ge 0$，$0\le d<l$），则 $a-j=d\in[0, l-1]$，$j$ 落在 $a$ 的窗口 $[a-l,\,a]$ 内，即 $j\in A^{(1)}_a$；$j\le a\le i$ 保持因果。窗口 $\{i-l,\dots,i\}$ 共 $l+1$ 个连续位置，对模 $l$ 的每个剩余类至少含一个代表。$i<l$ 时 $j$ 处于窗口内，一步直达。步数与 $n$ 无关，因此 $O(n\sqrt{n})$ 保持全局连通。Fixed 同样两步连通：同块内 $j\in A^{(1)}_i$ 一步直达（源自集合定义）；跨块时取目标块末尾的摘要单元 $s\ge j$，则 $j\in A^{(1)}_s$（同块且因果）且 $s\in A^{(2)}_i$（摘要列），两步到达[INFERENCE：按 §4.2 判据对 §4.3 集合的实例化]。

每位置依赖集大小（下标 0 起，为两端合计上界）：strided 为 $(l{+}1)+(\lfloor i/l\rfloor{+}1)$，fixed 为 $l+(c{+}1)(\lfloor i/l\rfloor{+}1)$；两端集合在窗口与同列、块内与摘要列重叠，去重后略小。取 $l\approx\sqrt{n}$ 时，两者每位置均为 $O(\sqrt{n})$，总量 $O(n\sqrt{n})$；fixed 多常数因子 $c$（§4.2–4.3）。

## 与残差块的组合、参数不变

分解头接入网络有三种方式（§5.1，式 6–8）：按残差块序号交错 $A^{(r\bmod p)}$；merged head 把 $\bigcup_m A^{(m)}$ 拼成一个头一次算完（仅多出常数因子）；标准多头 $n_h$ 路并行。参数量不随 $n_h$ 变：attend 内部权重矩阵维度按 $1/n_h$ 收缩，总参数不变（§5.1）。论文仅在此处论证参数不变，全文未采用 "fast weights" 或跨块缓存 K/V；跨块复用的是注意力模式定义（交错或合并），而非缓存张量。

跨块训练靠两件事撑住（§5.2）：pre-activation 残差使 $H_N$ 成为 $N$ 个子层函数之和，每个子层都从输出层直接获得梯度；$W_2$ 与 $W_p$ 初始化乘 $1/\sqrt{2N}$，残差块尺度对深度 $N$ 不变。省显存靠反向重算（§5.4）：注意力权重与前馈激活只在反向时重算，dropout 从注意力内部挪到每段残差相加之后（Figure 4）。

## 数据流与维度

形状记作 $[\text{行}, \text{列}]$。维度符号分别为：$n$ 序列长度；$d$ 模型维（论文取 $\{256, 512, 1024\}$）；$l$ stride（128/256）；$c$ fixed 模式每块摘要单元数（8/16/32）；$p=2$ 分解头数；$n_h$ 并行头数；$v$ 词表大小（字节级 256）；$n_{emb}$ 位置嵌入个数（图像 3、文本/音频 2）；$m_j$ 第 $j$ 个位置维大小。论文按单头书写，多头展开处标 [INFERENCE]。

### Stage 1 · 嵌入字节与位置

$$\begin{aligned}
E &= X W_e + \sum_{j=1}^{n_{emb}} O^{(j)} W_j && \text{// } [n, \bcancel{v}]\times[\bcancel{v}, d] + \textstyle\sum_j [n, \bcancel{m_j}]\times[\bcancel{m_j}, d] \to [n, d]\ \text{——字节嵌入加 } n_{emb} \text{ 个位置嵌入（式 15）}
\end{aligned}$$

图像用数据嵌入（行、列、通道，$n_{emb}=3$），文本/音频用注意力结构嵌入（宽 $l$ 矩阵的行号与列号，$n_{emb}=2$）；这类结构化嵌入对性能重要（§5.3）。

### Stage 2 · Strided 头：依赖集与 attend

$$\begin{aligned}
A^{(1)}_i &= \{t, \dots, i\},\ t=\max(0, i-l) && \text{// } |A^{(1)}_i|\le l{+}1\ \text{——局部窗口，} l{+}1 \text{ 个连续位置覆盖全部模 } l \text{ 剩余类}\\
A^{(2)}_i &= \{j : (i-j)\bmod l = 0\} && \text{// } |A^{(2)}_i|=\lfloor i/l\rfloor{+}1\ \text{——与 } i \text{ 同列的全部历史}\\
K_{S_i}, V_{S_i} &= \{W_k x_j\}_{j\in S_i},\ \{W_v x_j\}_{j\in S_i} && \text{// } [d, n] \xrightarrow{\text{按 } S_i \text{ 索引列}} [d, |S_i|]\ \text{——只 gather 依赖集内的列}\\
\beta_i &= (W_q x_i)\,K_{S_i}/\sqrt{d} && \text{// } [1, \bcancel{d}]\times[\bcancel{d}, |S_i|]\to[1, |S_i|]\ \text{——query 只对 } |S_i| \text{ 个 key 打分（式 3–4）}\\
a(x_i, S_i) &= V_{S_i}\operatorname{softmax}(\beta_i)^{\top} && \text{// } [d, \bcancel{|S_i|}]\times[\bcancel{|S_i|}, 1] \xrightarrow{\text{softmax 沿 } |S_i| \text{ 维}} [d, 1]\ \text{——依赖集内加权求和}
\end{aligned}$$

$S_i = A^{(m)}_i$，因果性由 $A^{(m)}_i\subseteq\{j\le i\}$ 保证。

### Stage 3 · Fixed 头：块内互看 + 摘要广播

$$\begin{aligned}
A^{(1)}_i &= \{j : \lfloor j/l\rfloor = \lfloor i/l\rfloor\} && \text{// } |A^{(1)}_i|\le l\ \text{——同块位置自由互看}\\
A^{(2)}_i &= \{j : j\bmod l \in \{l{-}c, \dots, l\}\} && \text{// } |A^{(2)}_i|\approx(c{+}1)(\lfloor i/l\rfloor{+}1)\ \text{——各历史块末尾的摘要单元}
\end{aligned}$$

多头 $n_h$ 时 attend 内维按 $1/n_h$ 收缩，参数不变；fixed 模式下各头看块内长度 $c$ 的不同子块（§4.3、§5.1）。

### Stage 4 · Pre-activation 残差块与前馈

$$\begin{aligned}
a(H) &= \operatorname{dropout}(\operatorname{attention}(\operatorname{norm}(H))) && \text{// } [n,d]\to[n,d]\ \text{——归一化后进稀疏注意力（式 12）}\\
b(H) &= \operatorname{dropout}(\operatorname{ff}(\operatorname{norm}(H + a(H)))) && \text{// } [n,d]\to[n,d]\ \text{——残差中间量归一化后进前馈（式 13）}\\
H_k &= H_{k-1} + a(H_{k-1}) + b(H_{k-1}) && \text{// } [n,d] \text{ 不变形}\ \text{——残差和始终保留，每个子层从输出直得梯度（式 10、14）}\\
ff(x) &= W_2 f(W_1 x + b_1) + b_2 && \text{// } [d, \bcancel{4d}]\times[\bcancel{4d}, d]\to[d]\ \text{——内层 } 4d\text{（half-size 时 } 2d\text{），} f(X)=X\operatorname{sigmoid}(1.702X)\\
y &= \operatorname{softmax}(\operatorname{norm}(H_N)\,W_{out}) && \text{// } [n, \bcancel{d}]\times[\bcancel{d}, v]\to[n, v]\ \text{——出口归一化后投影到词表（式 11）}
\end{aligned}$$

初始化（§5.2、§6）：token 嵌入 $N(0,\ 0.125/\sqrt{d})$，位置嵌入 $N(0,\ 0.125/\sqrt{d\,n_{emb}})$，注意力与前馈内权重 $N(0,\ 0.125/\sqrt{d_{in}})$，$W_2$ 与 $W_p$ 额外乘 $1/\sqrt{2N}$。

### Stage 5 · 训练与推理的实现差异

反向重算（§5.4，Figure 4 阴影为 checkpoint 张量）：注意力权重与 FFN 激活不驻留显存、在反向时重算，即可在 16,384 长度训练数百层 dense attention。block-sparse kernel（§5.5）：局部窗口分块计算、strided 转置后计算局部窗口、fixed 按块聚合；softmax 融合至单个 kernel，上三角不计算，省去负偏置项，操作数减半。混合精度（§5.6）：权重 FP32、激活与梯度 FP16，采样时 query-key 乘积切回 FP32 防半精度溢出。

## 实验证据

Table 1（bits per dim/byte，均为密度建模）：

| 任务 | 配置 | 参数 | 结果 | 此前最优 |
|---|---|---|---|---|
| CIFAR-10（3072 字节序列） | strided，128 层/2 头/$d{=}256$，half-size FFN 与 qk | 59M | 2.80（$\pm0.004$，§7.1） | PixelSNAIL 2.85 |
| Enwik8（上下文 12,288） | fixed，30 层/8 头/$d{=}512$，stride 128、$c{=}32$、merged | 95M | 0.99（$\pm0.001$，§7.2） | Transformer-XL 277M 0.99 |
| ImageNet 64×64 | strided，48 层/16 头/$d{=}512$，stride 128 | 152M | 3.44（单次，§7.3） | SPN 150M 3.52 |
| 古典音乐 12kHz µ-law | strided，152M | 152M | 1.97（Table 1、Table 4） | 无直接对照 |

Table 2 同上下文 dense 对照（bits per byte / 相对每步时间）：Enwik8 12,288 下 dense 1.00/1.31、fixed 0.99/0.55、strided 1.13/0.35；CIFAR-10 3,072 下 dense 2.82/0.54、fixed 2.85/0.47、strided 2.80/0.38。针对稀疏既更快又收敛到更低误差，论文提出两种未裁决的并列假说：分解模式的归纳偏置，或 full attention 自身的优化问题（§7、Table 2）。

Table 3 显示 Enwik8 随评估上下文加长单调改善：6,144→0.9952、9,216→0.9936、10,752→0.9932、11,904→0.9930、12,096→0.9922、12,160→0.9908。Table 4（音频序列长度换容量）：65,536/152M/1.97、262,144/25M/2.17、1,048,576/3M/2.99；§7.4 给出换算：长度 ×4 需容量约 ÷$4\sqrt{4}=8$。

训练配方（§6–7）：Adam、5000 步线性预热、梯度裁剪 1.0、权重衰减 0.01、余弦退火；Enwik8 80 epochs、dropout 0.40；CIFAR-10 120 epochs、lr 0.00035、dropout 0.25；ImageNet-64 70 epochs、dropout 0.01、64 块 V100 训 7 天。

## 边界

验证仅覆盖 CIFAR-10、ImageNet-64、Enwik8 与 µ-law 音频的密度建模（字节/维度似然），无机器翻译与摘要，全文无 BLEU 类指标。strided 要求数据与 stride 对齐，在文本上失败（Table 2 Enwik8 1.13），需换 fixed。音频超过 65,536 步后样本质量因容量缩减迅速下降（§7.4）；百万步序列模型只有 3M 参数，"length one million or more" 属原理演示（摘要）。dense 对照仅 Table 2 两处（Enwik8、CIFAR-10），论文并列两种假说且未裁决「稀疏更好」的成因。

## 相关页面

[[logsparse-self-attention]] 即 LogTrans 的 LogSparse，同属数据无关固定模式，以指数间隔与连通性定理面向时序预测；[[probsparse-self-attention]] 即 Informer 的 ProbSparse，由数据驱动挑选 query，与本文固定模式相对；[[native-sparse-attention]]、[[deepseek-sparse-attention]] 属后续硬件对齐稀疏注意力；[[fast-weight-mlp-view-of-attention]] 涉及 attention 的 fast-weight MLP 视角（本文未用该术语）。

[^src-child-2019-sparse-transformer]: [[source-child-2019-sparse-transformer]]
