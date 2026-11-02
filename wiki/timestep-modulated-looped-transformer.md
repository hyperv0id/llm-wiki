---
title: "Timestep-Modulated Looped Transformer (TMLT)"
type: technique
tags:
  - looped-transformer
  - architecture
  - weight-tying
  - timestep-encoding
  - reasoning
created: 2026-10-28
last_updated: 2026-10-28
source_count: 1
confidence: medium
status: active
---

# Timestep-Modulated Looped Transformer (TMLT)

TMLT 是 Xu & Sato（ICML 2025）为修补 [[looped-transformer-approximation-rate|looped Transformer 逼近率]] 里多出的两项 modulus 而提出的架构：权重仍共享，但每轮循环的归一化增益与子层输出乘上一组由 timestep 编码生成的缩放向量[^src-xu-2025-looped-transformers-timestep-encoding]。提出者是同一篇论文——理论先指出「同一个 FF 迭代 $K-1$ 次，每步增量由同一组权重决定」，架构再把逐 loop 的缩放补回去。

## 问题定位

逼近率里的两项受限 modulus 来自 weight-tied feed-forward 网络（§4.1）[^src-xu-2025-looped-transformers-timestep-encoding]。Lemma 4.1 的形式把病因写清楚了：给定目标向量序列 $y_k$，宽度 $18d$ 的 FF 层迭代 $K-1$ 次只能保证

$$
\left|\left(L_2\circ(\mathrm{id}+\mathrm{FF})^{\circ(K-1)}\circ L_1(k)-y_k\right)_i\right|\le\varepsilon_i ,
$$

误差上界就是相邻两步差 $\varepsilon_i$ 本身[^src-xu-2025-looped-transformers-timestep-encoding]。作者把问题重述为「式 18 的不等号能不能换成等号」[^src-xu-2025-looped-transformers-timestep-encoding]。

修法是给每轮一个缩放向量：$\mathrm{FF}(X)\to\eta(t)\odot\mathrm{FF}(X)$，$t$ 为 loop 下标，并按

$$
(\mathrm{id}+\eta\odot\mathrm{FF})^{r}:=(\mathrm{id}+\eta(r)\odot\mathrm{FF})\circ\cdots\circ(\mathrm{id}+\eta(1)\odot\mathrm{FF})
$$

定义复合[^src-xu-2025-looped-transformers-timestep-encoding]。逐 loop 存 $\eta(t)$ 会让参数量随 $r$ 线性增长，所以实际做法是用 timestep 编码经一个共享 MLP 生成这些参数；这个生成方式作者称为类比 Hypernetworks[^src-xu-2025-looped-transformers-timestep-encoding]。

## 机制

调制只作用在 scale 类参数上，attention 与 FFN 内部的权重不随 $t$ 变，仍是 weight-tied[^src-xu-2025-looped-transformers-timestep-encoding]。

**timestep 编码**：正弦频率编码经两层 MLP（隐层宽 $d$、SiLU）得到 $\mathrm{TE}(t)\in\mathbb{R}^d$：

$$
\begin{aligned}
\mathrm{PE}(t)_{2i}&=\sin\!\left(t/10000^{2i/d}\right),\qquad \mathrm{PE}(t)_{2i+1}=\cos\!\left(t/10000^{2i/d}\right)\\
\mathrm{TE}(t)&=W_3\cdot\mathrm{SiLU}\!\left(W_4\cdot\mathrm{PE}(t)+b_4\right)+b_3
\end{aligned}
$$

$W_3,W_4\in\mathbb{R}^{d\times d}$，$b_3,b_4\in\mathbb{R}^d$；$t$ 是 loop 序号，不是 token 位置（§4.2）[^src-xu-2025-looped-transformers-timestep-encoding]。

**四个调制向量**：一次线性映射出 $\alpha_1(t),\alpha_2(t),\gamma_1(t),\gamma_2(t)$，

$$
\alpha_1(t),\alpha_2(t),\gamma_1(t),\gamma_2(t)=W_5\cdot\mathrm{SiLU}\!\left(\mathrm{TE}(t)\right)+b_5,\qquad W_5\in\mathbb{R}^{4d\times d}.
$$

原文印的是 $b_5\in\mathbb{R}^{d}$，这里维度对不上：$W_5\mathrm{TE}(t)$ 输出 $4d$ 维，按行切成四个 $[d]$ 向量，逐元素相加要求 $b_5\in\mathbb{R}^{4d}$，按原文的 $\mathbb{R}^{d}$ 该式无法对齐（同一处的不自洽另记在「口径备注」）[^src-xu-2025-looped-transformers-timestep-encoding]。Theorem 4.2 的陈述直接给出 $\eta(t)\in\mathbb{R}^{4d}$，不经过 $b_5$，所以「理论保证」一节不受这一处印刷影响。

**带调制的 block**：归一化换成 time-dependent RMSNorm（[[rmsnorm|RMSNorm]]，Zhang & Sennrich, 2019），

$$
\mathrm{RMSN}(x,t)=\alpha(t)\odot\frac{x}{\mathrm{RMS}(x)},\qquad \mathrm{RMS}(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2},
$$

$$
\begin{aligned}
X'&=X+\gamma_1(t)\odot\mathrm{Attn}\!\left(\mathrm{RMSN}_1(X,t)\right)\\
\mathrm{TF}(X,t)&=X'+\gamma_2(t)\odot\mathrm{FF}\!\left(\mathrm{RMSN}_2(X',t)\right)
\end{aligned}
$$

$\gamma_1(t),\gamma_2(t)\in\mathbb{R}^d$ 逐 token 作用在子层输出上，残差支路本身不加权（§4.2）[^src-xu-2025-looped-transformers-timestep-encoding]。

## 数据流与维度

**维度符号**：$N$ 为序列长度（token 数）；$d$ 为模型维；$q$ 为 FFN 宽度；$r$ 为循环轮数；$t\in\{1,\dots,r\}$ 为循环下标。形状记作 $[\text{行},\text{列}]$；$\mathrm{PE}(t)\in\mathbb{R}^d$，下标 $2i$ 与 $2i+1$ 覆盖全部 $d$ 维；多头展开按单头书写（与论文一致）。

### Stage 1 · 把循环下标变成向量

$$
\begin{aligned}
\mathrm{PE}(t)_{2i},\ \mathrm{PE}(t)_{2i+1} &= \sin\!\left(t/10000^{2i/d}\right),\ \cos\!\left(t/10000^{2i/d}\right) && \text{// } [1]\to[d]\ \text{——标量 } t \text{ 摊成 } d/2 \text{ 对正余弦，频率随维数指数衰减}\\
\mathrm{TE}(t) &= W_3\,\mathrm{SiLU}\!\left(W_4\mathrm{PE}(t)+b_4\right)+b_3 && \text{// } [d]\xrightarrow{W_4}[d]\xrightarrow{\text{SiLU}}[d]\xrightarrow{W_3}[d]\ \text{——两层 MLP 把频率编码揉成可用的条件向量}
\end{aligned}
$$

### Stage 2 · 一次生成四个调制向量

$$
\begin{aligned}
\left[\alpha_1;\alpha_2;\gamma_1;\gamma_2\right](t) &= W_5\,\mathrm{SiLU}\!\left(\mathrm{TE}(t)\right)+b_5 && \text{// } [4d,\bcancel{d}]\times[\bcancel{d},1]\to[4d]\ \text{——消去 } d\text{，输出按行切成四个 } [d] \text{ 向量，对全部 token 共用}
\end{aligned}
$$

四个向量对所有 token 共用，靠广播进入下一阶段；参数量与 $r$ 无关，这是相对于逐 loop 存 $\eta(t)$ 的取舍[^src-xu-2025-looped-transformers-timestep-encoding]。

### Stage 3 · 第 $t$ 轮：先归一化再加调制

$$
\begin{aligned}
\bar{X} &= \alpha_1(t)\odot\frac{X}{\mathrm{RMS}(X)} && \text{// } [N,d]\xrightarrow{\text{RMS 沿 } d \text{ 维}}[N,1]\xrightarrow{\text{广播 }\alpha_1(t)\in[d]}[N,d]\ \text{——逐 token 缩放，不带 shift}\\
X' &= X+\gamma_1(t)\odot\mathrm{Attn}(\bar{X}) && \text{// } [N,d]\xrightarrow{\text{Attn}}[N,d]\xrightarrow{\gamma_1(t)\text{ 按 token 广播}}[N,d]\ \text{——注意力输出缩放后回残差}\\
X_{\mathrm{next}} &= X'+\gamma_2(t)\odot\mathrm{FF}\!\left(\mathrm{RMSN}_2(X',t)\right) && \text{// } [N,d]\xrightarrow{\text{FF 逐 token}}[N,d]\ \text{——FFN 内层 } q \text{ 宽，输出同样按 } \gamma_2(t) \text{ 缩放}
\end{aligned}
$$

### Stage 4 · 展开 $r$ 轮

$$
\begin{aligned}
X^{(t)} &= \mathrm{TF}\!\left(X^{(t-1)}+E,\ t\right),\quad t=1,\dots,r && \text{// } [N,d]\ \text{——同一组权重跑 } r \text{ 次；输入嵌入 } E \text{ 每轮重新加一次}
\end{aligned}
$$

论文只交代了「输入嵌入在每轮循环都加一次」与调制参数的初始化（$\gamma$ 取 0、$\alpha$ 取 1，出自 Peebles & Xie 2023 的做法，附录 C.1.3）[^src-xu-2025-looped-transformers-timestep-encoding]。$\gamma$ 初值为零意味着训练开始时调制支路等价于关掉，先学主干再学逐轮缩放。

## 理论保证

**Theorem 4.2**：给定 $y_k\in\mathbb{R}^d$、$k=0,\dots,K-1$，存在宽度 $6d$ 的 FF 层 $\mathrm{FF}:\mathbb{R}^{4d}\to\mathbb{R}^{4d}$、$\eta(t)\in\mathbb{R}^{4d}$（$t=1,\dots,K-1$）与两个仿射映射 $L_1:\mathbb{R}\to\mathbb{R}^{4d}$、$L_2:\mathbb{R}^{4d}\to\mathbb{R}^{d}$，使

$$
\left|\left(L_2\circ(\mathrm{id}+\eta\odot\mathrm{FF})^{\circ(K-1)}\circ L_1(k)-y_k\right)_i\right|=0
$$

对 $i=1,\dots,d$、$k=0,\dots,K-1$ 成立[^src-xu-2025-looped-transformers-timestep-encoding]。与 Lemma 4.1 比：误差从 $\le\varepsilon_i$ 变成 $=0$，工作维从 $12d$ 降到 $4d$，宽度从 $18d$ 降到 $6d$[^src-xu-2025-looped-transformers-timestep-encoding]。

口径要写清：Theorem 4.2 是关于 d 维向量逐点记忆的陈述，作用在 FF 层上；论文没有把它写成 Theorem 3.6 的替代版，也没有声称 TMLT 得到新的序列到序列 rate[^src-xu-2025-looped-transformers-timestep-encoding]。

## 证据

任务分两类：reasoning（LCS、Edit Distance、Sudoku、Countdown）考察 $r$ 与 timestep 编码的效果；核心 Transformer 基准是 in-context learning 与 WikiText-103 语言建模。reasoning 模型 4 头、256 维，TF 基线 LCS/ED 用 12 层、Sudoku/Countdown 用 6 层（附录 C.1.3）[^src-xu-2025-looped-transformers-timestep-encoding]。

**Table 2（测试精度）**：LCS/ED 各 10⁶ 训练 / 10³ 测试，Sudoku 3M / 100K，Countdown 5M / 1K（§5.1.1）[^src-xu-2025-looped-transformers-timestep-encoding]。

| Task | TF (L=6) | Looped r=4 | r=8 | r=16 | r=32 | w/ TE r=4 | r=8 | r=16 | r=32 |
|---|---|---|---|---|---|---|---|---|---|
| Sudoku | 0.0 | 0.0 | 0.0 | 65.6 | 87.9 | 0.0 | 0.0 | 62.0 | 90.2 |
| Countdown | 53.8 | 28.3 | 52.7 | 81.0 | 88.1 | 33.2 | 54.4 | 80.2 | 90.5 |

Table 2 下半块，TF 基线为 12 层，循环档位 5 / 10 / 50 / 100：

| Task | TF (L=12) | Looped r=5 | r=10 | r=50 | r=100 | w/ TE r=5 | r=10 | r=50 | r=100 |
|---|---|---|---|---|---|---|---|---|---|
| LCS (60) | 70.0 | 66.0 | 81.8 | 98.6 | 96.9 | 68.5 | 80.5 | 99.3 | 97.1 |
| LCS (100) | 39.8 | 39.6 | 45.1 | 93.5 | 98.2 | 36.7 | 45.6 | 98.1 | 98.6 |
| ED (40) | 54.2 | 41.4 | 57.9 | 85.4 | 90.4 | 44.8 | 63.5 | 94.5 | 96.1 |
| ED (60) | 41.4 | 23.8 | 32.6 | 47.3 | 47.7 | 26.6 | 38.9 | 57.3 | 88.3 |

表内无标准差、无种子数、无比参数列[^src-xu-2025-looped-transformers-timestep-encoding]。正文只单独点名 ED（长度 60）、$r=100$ 这一格（47.7 → 88.3）[^src-xu-2025-looped-transformers-timestep-encoding]。

三条读数：

1. **$r$ 是主要变量**：多数格子随 $r$ 单调升，$r=4/5$ 时 looped 模型仍不及 12 层或 6 层基线（ED (60) 23.8 vs 41.4；Sudoku $r\le8$ 全为 0.0）[^src-xu-2025-looped-transformers-timestep-encoding]。
2. **timestep 编码的增益在高 $r$ 处最大**：ED (60) $r=100$ 的 47.7 → 88.3；LCS (100) $r=50$ 的 93.5 → 98.1；ICL 测试 MSE 1.4e-02 → 1.7e-03，低于 12 层 TF 的 8.6e-03（Table 3）[^src-xu-2025-looped-transformers-timestep-encoding]。
3. **并非全胜**：Sudoku $r=16$（62.0 vs 65.6）、LCS (100) $r=5$（36.7 vs 39.6）加编码后更低；LCS (60) 在 $r=100$ 由 98.6 / 99.3 回落到 96.9 / 97.1。正文未讨论这些反向格[^src-xu-2025-looped-transformers-timestep-encoding]。

**Table 4（WikiText-103 困惑度）**：train 15.9 / 17.1 / 15.9，test 20.5 / 20.6 / 19.6，三列依次为 TF (L=12)、Looped $r=24$、w/ Timestep $r=24$[^src-xu-2025-looped-transformers-timestep-encoding]。语言模型按 GPT-2 decoder 搭，基线 6 层 8 头 512 维，looped 版 1 层 12 头 768 维，作者称按参数量对齐（附录 C.3）[^src-xu-2025-looped-transformers-timestep-encoding]。

**附录 D 的扰动实验**：WikiText-103 随机替换 10% token 后重训，加 timestep 编码的交叉熵 4.32 → 4.18，continuity coefficient 130.6 → 21.5；这组数字只在附录正文，无表号，未说明对应的 $r$、模型规模与方差[^src-xu-2025-looped-transformers-timestep-encoding]。

## 边界

- 理论只覆盖单层、固定长度、$r>N$；TMLT 本身的实验是 4 头 256 维的小模型加一个 768 维 GPT-2 规模的语言模型，不是大模型证据[^src-xu-2025-looped-transformers-timestep-encoding]。
- 用测试精度代理逼近能力，论据是附录 C.1.2 / Fig. 5 中训练与测试精度的强正相关（ED 长度 60）[^src-xu-2025-looped-transformers-timestep-encoding]。
- Table 2–4 都是单次结果，没有多种子统计；ED (60) 在 $r\le50$ 区间绝对值很低，只有加编码后的 $r=100$ 才拉开[^src-xu-2025-looped-transformers-timestep-encoding]。
- 论文没有给 TMLT 的参数量表，只说明逐 loop 存参数会随 $r$ 线性增长，故改用共享生成器[^src-xu-2025-looped-transformers-timestep-encoding]。
- 记忆与泛化的分工不在论文范围内：Theorem 4.2 是逐点记忆的结果，训练/测试相关性靠实验佐证，不是定理结论[^src-xu-2025-looped-transformers-timestep-encoding]。

## 口径备注

- 原文 $W_5\in\mathbb{R}^{4d\times d}$ 但写 $b_5\in\mathbb{R}^{d}$；按 $4d$ 维输出应为 $\mathbb{R}^{4d}$。本页照录原文符号，不改正[^src-xu-2025-looped-transformers-timestep-encoding]。
- 附录 C.1.3 写通用 batch size 64，C.3 写语言建模 batch size 48，两处不一致，本页分别引用[^src-xu-2025-looped-transformers-timestep-encoding]。
- 两处都记作 $\eta(t)$ 但维度不同：§4.1 是 $\mathbb{R}^d$ 的 FF 缩放向量，Theorem 4.2 是 $\mathbb{R}^{4d}$（对应 $4d$ 宽的工作空间）的缩放向量，论文未解释两者的对应关系；本页按各自的原文范围引用。

## 相关页面

理论来源见 [[looped-transformer-approximation-rate]]；这条路线在语言模型上的大规模实例见 [[recurrent-depth]]（Huginn，3.5B/800B token）；权重沿深度循环的源头见 [[universal-transformer]]；归一化细节见 [[rmsnorm]]，频率编码见 [[sinusoidal-positional-encoding]]。

[^src-xu-2025-looped-transformers-timestep-encoding]: [[source-xu-2025-looped-transformers-timestep-encoding]]
