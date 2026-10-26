---
title: "SDPA Output Gating（注意力输出门控）"
type: technique
tags:
  - attention
  - gating
  - llm
  - training-stability
  - attention-sink
  - long-context
created: 2026-10-26
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# SDPA Output Gating（注意力输出门控）

在 softmax attention 的 SDPA 输出侧、输出投影 $W_O$ 之前，乘一个由该 token 隐状态线性变换加 sigmoid 得到的门控 $G=\sigma(XW_\theta)$：位置固定在 $W_O$ 之前，粒度按 attention head 独立、逐元素生效，$X$ 取 pre-norm 之后的层输入，故门控随 query token 变化（式 5、Fig. 1；arXiv:2505.06708）[^src-gated-attention-for-large-language-models-2505-06708]。Qwen 团队在 15B MoE（15A2B）与 1.7B dense 上扫 30 个变体，把位置、粒度、head 独立性、乘性/加性、激活函数五维分开消融，结论是这一处乘法门控稳定优于所有对照；参数增量逐元素 201M、headwise 1.6M（§3.2.1 称 headwise 不足 2M）。

## 问题定位

门控在 LSTM、Highway、SSM、线性注意力里普遍存在，但少有工作单独度量它在 softmax attention 中的贡献：Switch Heads、NSA、MoSA 把门控与专家路由、稀疏设计捆绑，整体增益无法拆到门控这一项[^src-gated-attention-for-large-language-models-2505-06708]。Switch Heads 的消融给出直接线索——把 key/value 专家数压到 1 个、路由退化为只调制 value 输出（`Switch v, 1top1`，加 3M 参数），PPL 5.808 与基准 6.026 相比增益仍在，且优于专家更多的 `switch kv, 8top8`（加 38M，5.847），说明起作用的是门控本身而非路由（附录 A.1、表 6）[^src-gated-attention-for-large-language-models-2505-06708]。

## 机制

门控的通用形式（式 5）[^src-gated-attention-for-large-language-models-2505-06708]：

$$Y'=g(Y,X,W_\theta,\sigma)=Y\odot\sigma(XW_\theta)$$

$Y$ 被调制，$X$ 用来造门控分数，$W_\theta$ 为门控参数，$\sigma$ 默认 sigmoid。插入位置五处（Fig. 1 左）：$G_1$ 在 SDPA 输出之后，$G_2/G_3/G_4$ 分别在 V/K/Q 投影之后，$G_5$ 在拼接输出经 $W_O$ 之后[^src-gated-attention-for-large-language-models-2505-06708]。粒度两档：headwise 每个 head 一个标量，elementwise 与 $Y$ 同形逐通道；head 独立性两选：head-specific 每头一套 $W_\theta$，head-shared 把逐元素分数沿 $q$ 维平均成 $n\times d_k$ 一份；乘性 $Y'=Y\cdot\sigma(X\theta)$ 与加性 $Y'=Y+\sigma(X\theta)$ 两选，加性配 SiLU（sigmoid 输出限在 $[0,1]$、加性下不合适）；此外用 Identity 与 RMSNorm 替换门控以单独隔离非线性的作用[^src-gated-attention-for-large-language-models-2505-06708]。

## 数据流与维度变换

**维度符号**：$n$ 序列长度；$d_\text{model}=2048$ 隐维；$q=32$ query head 数；$k=4$ KV head 数；$d_k=128$ 头维；24 层。模型超参由表 1 的参数量反解（见下 Stage 2 后的核对），论文正文未直接给 $d_\text{model}$ 与层数，标 [INFERENCE]。形状记作 $[\text{行},\text{列}]$。

### Stage 1 · 标准 SDPA 得各头输出并拼接

$$\begin{aligned}
S^{(i)} &= \operatorname{softmax}\!\left(Q_iK_i^{\top}/\sqrt{d_k}\right) && \text{// } [n,\bcancel{d_k}]\times[\bcancel{d_k},n]\to[n,n]\ \text{——第 } i \text{ 个 head 的注意力图，每行对全序列归一}\\
H_i &= S^{(i)}V_i && \text{// } [n,\bcancel{n}]\times[\bcancel{n},d_k]\to[n,d_k]\ \text{——每个 token 取该头 } d_k \text{ 维上下文聚合}\\
O &= \operatorname{Concat}\!\left(H_1,\dots,H_q\right) && \text{// } q\times[n,d_k]\to[n,qd_k]\ \text{——} q=32 \text{ 头拼接，} qd_k=4096
\end{aligned}$$

### Stage 2 · 由层输入造门控

门控读的是 $X\in\mathbb{R}^{n\times d_\text{model}}$（该层 pre-norm 后的输入，与 Q/K/V 投影同源），不读 $O$；$W_\theta\in\mathbb{R}^{d_\text{model}\times qd_k}$[^src-gated-attention-for-large-language-models-2505-06708]。

$$\begin{aligned}
XW_\theta &\in\mathbb{R}^{n\times qd_k} && \text{// } [n,\bcancel{d_\text{model}}]\times[\bcancel{d_\text{model}},qd_k]\to[n,qd_k]\ \text{——同一 token 的隐状态展开成 } q\times d_k \text{ 个门控 logit}\\
G &= \sigma\!\left(XW_\theta\right),\ \sigma=\frac{1}{1+e^{-x}} && \text{// } [n,qd_k]\xrightarrow{\text{逐元素 sigmoid}}[0,1]^{n\times q\times d_k}\ \text{——逐 token、逐 head、逐通道各取一个独立权重}
\end{aligned}$$

门控形状即 $n\times q\times d_k$：**逐 token**（每个 query 位置一套）、**逐 head**（$q=32$ 套互不共享）、**逐通道**（$d_k=128$ 个通道各自取值）。参数增量为每层 $d_\text{model}\cdot q\cdot d_k$[^src-gated-attention-for-large-language-models-2505-06708]（按行数与表 1 反解核对：$2048\times32\times128=8.39$M/层，24 层 $=201$M，表 1 第 5 行 201M；headwise 变体 $W_\theta\in\mathbb{R}^{d_\text{model}\times q}$，$2048\times32=65.5$k/层，24 层 $=1.57$M，表 1 第 10 行 1.6M；$G_5$ 位置 $2048^2\times24=100.7$M，表 1 第 9 行 100M [INFERENCE]）。

### Stage 3 · 门控逐元素相乘

$$\begin{aligned}
O' &= O\odot G && \text{// } [n,q,d_k]\odot[n,q,d_k]\to[n,q,d_k]\ \text{——形状不变，对角重加权；因 } G \text{ 只依赖 } X_i \text{，权重随 query 变、与 key 无关}
\end{aligned}$$

### Stage 4 · 输出投影

$$\begin{aligned}
Y &= O'W_O && \text{// } [n,\bcancel{qd_k}]\times[\bcancel{qd_k},d_\text{model}]\to[n,d_\text{model}]\ \text{——回到残差流；门控在此乘法之前完成}
\end{aligned}$$

$G_1$ 相对 $G_2$ 的差别只在门控支路读哪个张量：$G_2$ 的 $W_\theta$ 造出 $n\times k\times d_k$ 的分数作用于 $XW_V$，因此依赖位置 $j$（key/value 侧）而非当前 query；$G_3/G_4$ 同理落在 K/Q 投影后；$G_5$ 落在 $W_O$ 之后。门控默认无归一化，sigmoid 输出直接乘上；加性变体把乘改成加并换 SiLU[^src-gated-attention-for-large-language-models-2505-06708]。

## 机理：一段低秩映射被逐元素非线性打断

### 代数链条

单头输出可把输出投影并进 value 投影（式 6）[^src-gated-attention-for-large-language-models-2505-06708]：

$$\begin{aligned}
o_i^{c} &= \left(\sum_j S^{c}_{ij}\,X_jW_V^{c}\right)W_O^{c} = \sum_j S^{c}_{ij}\,X_j\left(W_V^{c}W_O^{c}\right) && \text{// } [d_\text{model},\bcancel{d_k}]\times[\bcancel{d_k},d_\text{model}]\to[d_\text{model},d_\text{model}]\ \text{——两段线性合成一个秩 } \le d_k \text{ 的映射}
\end{aligned}$$

$S^c_{ij}$ 是第 $c$ 头内 token $i$ 对 $j$ 的注意力分数，$X_jW_V^c$ 是 $j$ 的 value 输出；$d_k=128<d_\text{model}=2048$ 时该合成映射秩受限，GQA 组内共享 $W_V$ 进一步压低各头的差异[^src-gated-attention-for-large-language-models-2505-06708]。两个修改对应两种打断方式（式 7、式 8）：$G_2$ 把非线性放在每个 $X_jW_V^c$ 上（作用于被聚合的 value，与 query 无关），$G_1$ 把非线性放在聚合之后（作用于当前 query 的混合结果）[^src-gated-attention-for-large-language-models-2505-06708]。$G_1$ 的具体形式即对角门控嵌进两段线性之间：

$$\begin{aligned}
o_i^{c} &= \operatorname{NLM}\!\left(\sum_j S^{c}_{ij}\,X_jW_V^{c}\right)W_O^{c},\quad \operatorname{NLM}(z)=z\odot\sigma\!\left(X_iW_\theta^{c}\right) && \text{// } W_V^{c} \text{ 与 } W_O^{c} \text{ 之间不能再合并；} G \text{ 依赖 } X_i \text{ 而非 } X_j
\end{aligned}$$

$G_5$ 加在 $W_O$ 之后，两段线性已在前面合并完毕，故无增益（表 1 第 9 行 PPL 6.017，基准 6.026）[^src-gated-attention-for-large-language-models-2505-06708]。这条代数链在表 3 上有一组可核对的对照：替换为 RMSNorm（每头归一化，几乎零参数）5.847、只加 SiLU 不加参数 5.975、加性门控去掉 SiLU 5.882、加性门控配 SiLU 5.821，四者都优于基准 6.026、都弱于乘性 sigmoid 门控 5.761——非线性本身即有增益，乘性 sigmoid 在此之上再加稀疏[^src-gated-attention-for-large-language-models-2505-06708]。RMSNorm 这一支与 [[rmsnorm]] 同源，差别是作用在拼接前的每个 head 输出上。

### 稀疏：门控分数集中在 0 附近

门控均值与 PPL 在表 4 上大体成反序，例外是 head-shared 与 NS-sigmoid 两行：$G_1$ 逐元素 0.116→5.761、$G_1$ headwise 0.172→5.792、$G_2$ 逐元素 0.221→5.820、$G_1$ head-shared 0.271→5.801、$G_1$ 与输入无关 0.335→5.917、NS-sigmoid 0.653→5.900[^src-gated-attention-for-large-language-models-2505-06708]。NS-sigmoid 即 $0.5+0.5\,\mathrm{sigmoid}(x)$，把分数压在 $[0.5,1]$：非线性保留、稀疏被去掉，增益从 0.265 PPL 掉到 0.126 PPL[^src-gated-attention-for-large-language-models-2505-06708]。逐元素 sigmoid 门控把 SDPA 输出的绝对均值从 0.71 压到 0.05（无门控基准 0.04），且在阈值 $10^{-2}$ 下的稀疏比例由 0.03 升到 0.44、$10^{-3}$ 下由 0.003 升到 0.126；把同样的小均值换成「乘平均门控值」只能到 0.33 / 0.080，说明额外稀疏来自门控分数的逐元素差异，而非整体缩放（附录 A.2、Fig. 4–5）[^src-gated-attention-for-large-language-models-2505-06708]。

### 定性解释：sink 与 massive activation（作者的假设，非推导）

作者把稀疏门控读成「按 query 过滤无关上下文」，从而顶替 attention sink 原本承担的角色；这一步是假设，论文未给出解析论证，Limitations 也承认 sink 到长上下文泛化之间缺严格理论[^src-gated-attention-for-large-language-models-2505-06708]。可核对的量测链条是：门控越稀疏，首 token 注意力占比（F-Attn）与层内最大激活（M-Act）越低——$G_1$ 逐元素 0.048 / 94，headwise 0.073 / 98，head-shared 0.301 / 286，$G_2$ 逐元素 0.297 / 125，输入无关 0.364 / 471，NS-sigmoid 0.451 / 892，基准 0.467 / 1053（表 4）[^src-gated-attention-for-large-language-models-2505-06708]。$G_2$ 一行是关键反例：大激活已被压到 125，sink 仍留在 0.297，说明大激活不是 sink 的必要条件；层内分析进一步显示基准的 massive activation 源自第 6 层 FFN 输出并经残差流传播，带 $G_1$ 门控的模型逐层缓慢增长、任何层都不出现明显 sink（附录 A.3）[^src-gated-attention-for-large-language-models-2505-06708]。训练稳定性被归到同一条链：大激活减少使 BF16 训练更少触发数值误差；但附录 A.5 的截断实验（把 attention/FFN 输出夹到 300 或 100）在 lr $8\times10^{-3}$ 下仍不收敛，作者据此承认大激活不是不稳定的唯一来源[^src-gated-attention-for-large-language-models-2505-06708]。

## 选择 sigmoid 与逐头粒度的证据

| 选择 | 对照（表 1 / 表 4 行号） | Avg PPL | MMLU | 门控均值 | F-Attn |
|---|---|---|---|---|---|
| sigmoid vs SiLU | 表 1 第 5 行 vs 第 15 行（同为 $G_1$ 逐元素、201M） | 5.761 vs 5.822 | 60.82 vs 60.49 | — | — |
| sigmoid vs NS-sigmoid | 表 4 第 2 行 vs 第 7 行 | 5.761 vs 5.900 | 60.82 vs 60.05 | 0.116 vs 0.653 | 0.048 vs 0.451 |
| 逐元素 vs headwise | 表 1 第 5 行 vs 第 10 行（201M vs 1.6M） | 5.761 vs 5.792 | 60.82 vs 60.05 | 0.116 vs 0.172 | 0.048 vs 0.073 |
| head-specific vs head-shared | 表 1 第 10 行（1.6M）vs 第 12 行（201M） | 5.792 vs 5.801 | 60.05 vs 60.06 | 0.172 vs 0.271 | 0.073 vs 0.301 |
| 乘性 vs 加性 | 表 1 第 5 行 vs 第 14 行（加性配 SiLU） | 5.761 vs 5.821 | 60.82 vs 60.06 | — | — |

表中数字为表 1、表 4 原值[^src-gated-attention-for-large-language-models-2505-06708]。

sigmoid 的优势落在稀疏：同为 $G_1$ 逐元素，把 SiLU 换成 sigmoid 得 0.061 PPL；而 NS-sigmoid 保住非线性、去掉稀疏，反而丢掉 0.139 PPL。粒度只差 0.031 PPL，参数却差 125 倍，headwise 是参数效率更好的配置。head-shared 与 head-specific 的 PPL 几乎相同（差 0.009），但前者的门控均值与 F-Attn 明显更差，即「各头要不同的稀疏度」这条论断由 F-Attn 与门控均值支撑、而非由 PPL 支撑[^src-gated-attention-for-large-language-models-2505-06708]。

## 消融网格与数字

### 15A2B MoE，400B token（表 1）

训练配置：warmup 到 lr $2\times10^{-3}$、cosine 衰减到 $3\times10^{-5}$、bsz 1024、100k 步、上下文 4096、$d_k=128$、基准 $q=32$/$k=4$[^src-gated-attention-for-large-language-models-2505-06708]。

| 变体 | Act | Score shape | 新增参数(M) | Avg PPL | Hellaswag | MMLU | GSM8k | C-eval |
|---|---|---|---|---|---|---|---|---|
| (1) 基准 | — | — | 0 | 6.026 | 73.07 | 58.79 | 52.92 | 60.26 |
| (2) $k=8$ | — | — | 50 | 5.979 | 73.51 | 59.78 | 52.16 | 62.26 |
| (3) $q=48$ | — | — | 201 | 5.953 | 73.59 | 58.45 | 53.30 | 59.67 |
| (4) 加 4 专家 | — | — | 400 | 5.964 | 73.19 | 58.84 | 52.54 | 63.19 |
| (5) $G_1$ 逐元素 | sigmoid | $n\times q\times d_k$ | 201 | 5.761 | 74.64 | 60.82 | 55.27 | 62.20 |
| (6) $G_2$ V 逐元素 | sigmoid | $n\times k\times d_k$ | 25 | 5.820 | 74.38 | 59.17 | 53.97 | 61.00 |
| (7) $G_3$ K 逐元素 | sigmoid | $n\times k\times d_k$ | 25 | 6.016 | 72.88 | 59.18 | 50.49 | 61.74 |
| (8) $G_4$ Q 逐元素 | sigmoid | $n\times q\times d_k$ | 201 | 5.981 | 73.01 | 58.74 | 53.97 | 62.14 |
| (9) $G_5$ dense 输出 | sigmoid | $n\times d_\text{model}$ | 100 | 6.017 | 73.32 | 59.41 | 50.87 | 59.43 |
| (10) $G_1$ headwise | sigmoid | $n\times q$ | 1.6 | 5.792 | 74.50 | 60.05 | 54.44 | 62.61 |
| (11) $G_2$ V headwise | sigmoid | $n\times q$ | 0.2 | 5.808 | 74.38 | 59.32 | 53.53 | 62.61 |
| (12) $G_1$ head-shared | sigmoid | $n\times d_k$ | 201 | 5.801 | 74.34 | 60.06 | 53.15 | 61.01 |
| (13) $G_2$ V head-shared | sigmoid | $n\times d_k$ | 25 | 5.867 | 74.10 | 59.02 | 53.03 | 60.61 |
| (14) $G_1$ 加性 | SiLU | $n\times q\times d_k$ | 201 | 5.821 | 74.81 | 60.06 | 53.30 | 60.98 |
| (15) $G_1$ 逐元素 | SiLU | $n\times q\times d_k$ | 201 | 5.822 | 74.22 | 60.49 | 54.59 | 62.34 |

所有数字为表 1 原值[^src-gated-attention-for-large-language-models-2505-06708]。位置是最大方差来源：$G_1$ 与 $G_2$ 有增益，$G_3/G_4/G_5$ 基本回落到基准附近；$G_1$ 逐元素门控以 201M 参数同时压过 $k=8$（50M，5.979）、$q=48$（201M，5.953）、加 4 专家（400M，5.964）三个参数扩张对照，PPL 差 0.19–0.22、MMLU 差 1.0–2.4 点。摘要里的「最多 0.2 PPL、2 点 MMLU」对应第 5 行与第 1 行。

### 1.7B dense：多配置与稳定性（表 2）

门控按 FFN 收窄保持同参数量。28 层 400B（bsz 1024）：基准 4.0e-3 时 PPL 7.499、MMLU 50.21，门控 7.404、51.15；28 层 3.5T（bsz 2048）：6.180→6.130，HumanEval 34.15→37.80；48 层 400B：基准 4.0e-3 为 7.421，抬到 8.0e-3 退化到 9.195（HumanEval 21.34、GSM8k 15.24），改配 sandwich norm 回到 7.407，与 4.0e-3 基准的 7.421 基本持平，而门控在 8.0e-3 下得 7.325（MMLU 54.47、GSM8k 36.62）；48 层 1T：基准 5.3e-3 为 7.363，8.0e-3 直接发散，门控在 8.0e-3 下得 7.078（MMLU 56.47、GSM8k 39.73）[^src-gated-attention-for-large-language-models-2505-06708]。稳定性这一列是「更大 lr/bsz 下不发散」，与 PPL 增益是两件事：3.5T 同 lr 下门控只赢 0.050 PPL。

### 长上下文（表 5，RULER）

3.5T token 模型先在 32k 上继续训练 80B token（RoPE base 10k→1M），再用 YaRN 外推到 128k[^src-gated-attention-for-large-language-models-2505-06708]。

| | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|---|
| 基准（32k 训练内） | 88.89 | 85.88 | 83.15 | 79.50 | — | — |
| 门控（32k 训练内） | 90.56 | 87.11 | 84.61 | 79.77 | — | — |
| 基准 + YaRN | 82.90(−6.0) | 71.52(−14.4) | 61.23(−21.9) | 37.94(−41.6) | 37.51 | 31.65 |
| 门控 + YaRN | 88.13(−2.4) | 80.01(−7.1) | 76.74(−7.9) | 72.88(−6.9) | 66.60 | 58.82 |

括号内是相对各自未外推分数的跌幅[^src-gated-attention-for-large-language-models-2505-06708]。训练长度内门控只领先 0.27 分（32k），外推后 16k 与 32k 处差距拉开到 15.5 与 34.9 分；收益集中在「改 RoPE base 后衰减更慢」，而非训练长度内的长上下文能力。

### 与 Switch Heads 的对照（附录 A.1、表 6）

`switch v, 1top1`（3M，等价于 $G_2$ headwise 门控）5.808 是表 6 内最好，`switch v, 4top4`（13M）5.820、`switch v, 8top2`（25M）5.870、`switch kv, 4top4`（13M）5.935、`switch kv, 8top8`（38M）5.847，基准 6.026[^src-gated-attention-for-large-language-models-2505-06708]。增加激活专家数未带来超过 1top1 的收益：kv 侧 4top4→8top8 的 PPL 只改善 0.088（5.935→5.847），v 侧 4top4→8top2 反而退化（5.820→5.870），PPL 与各基准的最好值都落在只调制 value 输出的 1top1 上，作者据此把 Switch Heads 的增益归给门控本身。

## 边界

- **增益是同配置对照**：三个参数扩张基线（$k=8$、$q=48$、加 4 专家）与门控模型共用同一数据、token 预算与训练配方，只能说明「同样的参数花在门控上比花在宽度/专家上更值」，不构成与其他架构改进的跨家族比较[^src-gated-attention-for-large-language-models-2505-06708]。
- **规模只覆盖两档**：15B MoE 与 1.7B dense，最大 3.5T token；层数与 $d_\text{model}$ 需由表 1 参数量反解才能得到[^src-gated-attention-for-large-language-models-2505-06708]。
- **位置结论依赖 GQA 设定**：$G_1/G_2$ 有增益的代数解释依赖 $W_V$、$W_O$ 可合并且 $d_k<d_\text{model}$，GQA 组内共享 $W_V$ 被列为压低表达力的进一步因素；MHA 或 $d_k=d_\text{model}$ 时该论据的强度未测[^src-gated-attention-for-large-language-models-2505-06708]。
- **长上下文的证据边界**：64k/128k 只有 YaRN 外推行有数字，训练长度内（4k–32k）门控与基准差距在 0.27–1.67 分之间；「10 分以上」的提升出现在外推之后[^src-gated-attention-for-large-language-models-2505-06708]。
- **实现开销**：每层多一次 $d_\text{model}\to qd_k$ 的 GEMM 与一次逐元素乘，wall-time 增幅小于 2%（§3.1，未给 kernel 级实现与显存开销）[^src-gated-attention-for-large-language-models-2505-06708]。
- **机理未闭合**：非线性的作用、sink 如何影响长上下文泛化都只有实验与假设；截断实验表明大激活不是不稳定唯一来源（Limitations、附录 A.5）[^src-gated-attention-for-large-language-models-2505-06708]。

## 相关页面

- [[attention-sink]] — 本页给出 sink 被输入相关稀疏门控消除的一手量测
- [[native-sparse-attention]] — 同为 sigmoid 门控，但那是三支路输出的**分支级**加权系数，不逐 head、不逐通道
- [[moba-attention]] — **块级** top-$k$ 门控决定读哪些 KV，与输出侧逐元素门控正交
- [[rmsnorm]] — 表 3 用作 $G_1$ 位置的零参数非线性对照
- [[gated-temporal-attention]] — 时间序列插补模型 SADI 内部的时间维自注意力残差模块，与本页机制无继承关系
- [[source-gated-attention-for-large-language-models-2505-06708]] — 源摘要

[^src-gated-attention-for-large-language-models-2505-06708]: [[source-gated-attention-for-large-language-models-2505-06708]]
