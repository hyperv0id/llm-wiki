---
title: "MoBA: Mixture of Block Attention"
type: technique
tags:
  - attention
  - sparse-attention
  - mixture-of-experts
  - long-context
  - llm
created: 2026-10-08
last_updated: 2026-10-27
source_count: 4
confidence: medium
status: active
---

# MoBA: Mixture of Block Attention

Moonshot AI（Kimi 团队）于 2025-02 提出长上下文注意力架构（arXiv:2502.13189），把 [[mixture-of-experts|MoE]] 的 top-k 门控原则用于 attention，已部署于 Kimi 的长上下文请求[^src-moba-mixture-of-block-attention]。

## 问题

标准 attention 的计算量随序列长度二次增长，是扩展 LLM 有效上下文的主要开销。MoBA 的出发点是既有两条路线各有代价[^src-moba-mixture-of-block-attention]：

一是强偏置（sink attention、sliding window attention），稀疏模式预先定义且特定于任务，无法适配内容相关的重要程度；二是线性 attention（Mamba、RWKV、RetNet 等），将 softmax 换成线性近似，与既有 Transformer 差异大，转换成本高或需从头训练，在复杂推理任务上的有效证据有限。

MoBA 的设计原则是 "less structure"：不预定义稀疏模式，让模型自主学习注意位置。

## 机制

### 块即专家，query 自任门控

上下文按块大小 $B$ 切成 $n$ 个块 $I_i$。对 query $\bm{q}$，块 $i$ 的亲和分数用 query 与块内 key 均值的内积计算：

$$s_i = \langle \bm{q},\ \mathrm{mean\_pool}(\bm{K}[I_i])\rangle$$

门控 $g_i \in \{0,1\}$ 取 top-k，query 只对 $g_i=1$ 的块做 attention。与标准 MoE 的两点差异：门控输入是 query 本身而非额外网络，不引入新参数；专家是 KV 块而非 FFN 子网络，切分沿上下文长度维度而非 FFN 中间维度[^src-moba-mixture-of-block-attention]。

### 因果性的两个设计

1. **不选未来块**：位置在 query 之后的块 $s_i = -\infty$、$g_i = 0$。
2. **当前块强制选择 + causal mask**：query 所在块的 mean pooling 会混入未来 token 的信息，因此当前块强制 $g_i=1$ 且块内 attention 加 causal mask，作用相当于 MoE 的 shared expert（静态规则叠加在动态路由上）[^src-moba-mixture-of-block-attention]。

### 与静态稀疏模式的关系

sliding window attention 与 attention sink（现象见 [[attention-sink]]）均为 MoBA 的特例：前者等价于门控始终选最近块，后者等价于始终选初始块与最近块；MoBA 的门控则随输入变化，不限于固定模式[^src-moba-mixture-of-block-attention]。

### 与 full attention 的无缝切换

MoBA 是 full attention 的 drop-in 替代：参数量不变，top-k 覆盖全部块时退化为 full attention；各层可在初始化或训练中动态选择这两种模式[^src-moba-mixture-of-block-attention]。

### 实现

五步流程为：门控加因果掩码确定 query 到块的分配；按分配块重排 query；用变长 FlashAttention 计算块级 attention（当前块 causal、历史块 non-causal 分开计算）；输出重排回原序；用 online softmax（tiling）合并跨块输出[^src-moba-mixture-of-block-attention]。

## 数据流与维度变换

**维度符号**：$N$ 序列长度；$B$ 块大小；$n=N/B$ 块数；$k$ top-$k$；$d_k$ 单头维度。形状记 $[\text{行},\text{列}]$，按论文 Algorithm 1 的单头口径书写，多头由各 head 独立计算后拼接[^src-moba-mixture-of-block-attention]。

### Stage 1 · 分块并 mean-pool 出块级 key

$$\begin{aligned}
\{\tilde{\bm{K}}_i,\tilde{\bm{V}}_i\} &= \mathrm{split\_blocks}(\bm{K},\bm{V},B) && \text{// } \bm{K},\bm{V}\in[N,d_k] \xrightarrow{\text{沿 } N \text{ 维切成 } n \text{ 段}} \tilde{\bm{K}}_i,\tilde{\bm{V}}_i\in[B,d_k]\ \text{——每个 KV 块当作一个专家，共 } n \text{ 个}\\
\bar{\bm{K}} &= \mathrm{mean\_pool}(\bm{K},B) && \text{// } [n,B,d_k] \xrightarrow{\text{沿 } B \text{ 维取均值}} [n,d_k]\ \text{——每块压成一个无参数 landmark key，}\bcancel{B}\text{ 消去}
\end{aligned}$$

### Stage 2 · 门控打分与块级 top-k

$$\begin{aligned}
\bm{S} &= \bm{Q}\bar{\bm{K}}^{\top} && \text{// } [N,\bcancel{d_k}]\times[\bcancel{d_k},n]\to[N,n]\ \text{——消去 } d_k\text{，每个 query 对 } n \text{ 个块各得一个亲和分数（式 6 的批量化）}\\
\bm{G} &= \mathrm{topk}(\bm{S}+\bm{M},k) && \text{// } [N,n] \xrightarrow{\text{沿 } n \text{ 维逐行 Top-}k} [N,k]\ \text{——每个 query 选 } k \text{ 个块（式 5）；}\bm{M}\text{ 为块级因果掩码：未来块置 } -\infty\text{（}g_i=0\text{），当前块强制入选（}g_i=1\text{）}
\end{aligned}$$

完整张量为 $\bm{S}\in[N,h,n]$：门控逐头计算，各 head 可选不同块。当前块占用一个名额，故历史块至多 $k-1$ 个（论文脚注：top-3 即至多 2 个历史块 + 当前块）[^src-moba-mixture-of-block-attention]。

### Stage 3 · 两段分开的块内 attention

**支路 A：当前块（causal）**

$$\begin{aligned}
\bm{O}^{s} &= \mathrm{flash\_attention\_varlen}(\bm{Q}^{s},\tilde{\bm{K}}^{s},\tilde{\bm{V}}^{s},\mathrm{causal=True}) && \text{// } \bm{Q}^{s}\in[N,d_k]\ \text{对各自所在块的}\ [B,d_k]\ \text{——块内加 token 级 causal mask，堵住 mean-pool 混入的未来信息}
\end{aligned}$$

**支路 B：选中历史块（non-causal）**

$$\begin{aligned}
\bm{O}^{m} &= \mathrm{flash\_attention\_varlen}(\bm{Q}^{m},\tilde{\bm{K}}^{m},\tilde{\bm{V}}^{m},\mathrm{causal=False}) && \text{// } \bm{Q}^{m}\ \text{按 } \bm{G} \text{ 重排后对选中历史块的}\ [B,d_k]\ \text{——历史块内 key 全在 query 之前，无需 mask}
\end{aligned}$$

### Stage 4 · online softmax 合并跨块输出

$$\begin{aligned}
\bm{O} &= \mathrm{combine\_with\_online\_softmax}(\bm{O}^{s},\bm{O}^{m}) && \text{// } [N,d_k]\ \text{的至多 } k \text{ 段输出} \xrightarrow{\text{按各段 log-sum-exp 重加权求和}} [N,d_k]\ \text{——严格等价于式 (2) 在 } I=\bigcup_{g_i>0} I_i \text{ 上的一次 softmax}
\end{aligned}$$

与 Stage 2 协调时，全部掩码均在 softmax 前生效：块级选择（$-\infty$ 填充 + top-$k$）作用于打分矩阵 $\bm{S}$，先于任何 softmax；当前块的 causal mask 在支路 A 段内 softmax 前生效，被淘汰块不进入 softmax 分母。

## 门控归一化与 full attention 的关系

根据 Stage 3–4，query $\bm{q}$ 的注意力权重只在选中块并集 $I=\bigcup_{g_i>0}I_i$ 内归一化（式 2、4）[^src-moba-mixture-of-block-attention]：

$$p_j=\frac{e^{\langle\bm{q},\bm{k}_j\rangle}}{\sum_{j'\in I}e^{\langle\bm{q},\bm{k}_{j'}\rangle}}\quad(j\in I),\qquad p_j=0\quad(j\notin I)$$

被淘汰块的概率严格为 0。成立条件有二：门控为硬选择（$g_i\in\{0,1\}$，式 5），且归一化分母定义为 $I$ 而非全上下文；淘汰块既不出现在分子（不参与计算）也不进入分母，直接从 softmax 支撑集中删除。online softmax 合并不改变该结论：各段局部 softmax 各自带分母，合并时按指数权重还原出 $I$ 上的同一分母。

与 full attention 的差别只在分母：

$$Z_{\mathrm{MoBA}}=\sum_{j\in I^{\circ}}e^{\langle\bm{q},\bm{k}_j\rangle}\ \leq\ Z_{\mathrm{full}}=\sum_{j\leq \mathrm{pos}(\bm{q})}e^{\langle\bm{q},\bm{k}_j\rangle}$$

其中 $I^{\circ}=I\cap[1,\mathrm{pos}(\bm{q})]$ 即 $I$ 去掉当前块内位于 query 之后的 token（由支路 A 的 causal mask 清零）。差额是被淘汰 token 的 $e^{\langle\bm{q},\bm{k}_j\rangle}$ 之和且通常非零，MoBA 并非 full attention 的无损近似。两者严格相等当且仅当 top-$k$ 覆盖全部因果允许块，即 $k\geq\lceil \mathrm{pos}(\bm{q})/B\rceil$（全局最坏 $k\geq n$），此时退化为 full attention。此外，选块分数（query 对块内 key 均值，式 6）不同于块内权重分数（query 对单个 key，式 2），前者仅是后者的块级代理。论文将 MoBA 定位为替代 full attention 训练（从零训练或继续预训练）的架构[^src-moba-mixture-of-block-attention]；受分母差额影响，训练后无法直接作为推理期近似。

## 证据

以下均出自论文实验，各条目写明设置。

### 缩放定律（8K 训练）

在 5 个模型（568M–2.1B，Chinchilla 足量 token）上，序列长 8K，块 512，top-3，稀疏度至多 $1-\frac{512\times 3}{8192}=81.25\%$（正文同段末尾写 sparsity up to 75%，与公式口径不一）。验证集 LM loss 与 full attention 的差保持在 1e-3 内；但在 32K 外推的 trailing loss 上，正文表格（seqlen=32K、last 2K）拟合为 MoBA $1.546\times C^{-0.108}$、full $1.464\times C^{-0.097}$，纯 MoBA 在尾部 token 上落后（图 3 标题写 last 1K、表格写 last 2K，文内口径不一）[^src-moba-mixture-of-block-attention]。

附录 A.1 按位置分段评测以消除自然数据偏短上下文的偏差：trailing token 贡献了稀疏与 full attention 的大部分差距，长上下文缩放研究可聚焦 trailing token[^src-moba-mixture-of-block-attention]。

### 块粒度消融

1.5B 模型、32K 上下文，块数 8 到 128（top-k 相应 2 到 32），稀疏度固定 75%。更细的块在同一稀疏度下损失更低，最粗设置（8 选 2）与更细设置间相差 1e-2[^src-moba-mixture-of-block-attention]。

### 混合训练与逐层混合

1.5B、30B token、32K 上下文（块 2048、top-3）[^src-moba-mixture-of-block-attention]：

MoBA 与 full 两阶段训练中，前 90% token 用 MoBA、后 10% 切至 full attention，position-wise loss 几乎回到 full 水平，切换时未观察到明显 loss spike。面向 SFT 的逐层混合设置下，纯 MoBA 的 SFT loss 偏高，因 SFT 的 loss masking（prompt 不计损失）只从回答 token 回传梯度，对稀疏 attention 构成稀疏梯度；把最后几层换成 full attention 可显著降低 SFT loss。

### 1M 上下文评测

基于 Llama 3.1 8B 持续预训练至 1M 上下文（256K 阶段起点用位置插值），MoBA 训练 100B token（块 4096、top-12，1M 时稀疏度至多 95.31%），最后 3 层保留 full attention、其余 29 层换 MoBA。评测中 MoBA 仅用于 prefill，生成期切回 full attention。RULER 128K（稀疏度 62.5%）0.7818 vs full 0.7849；GSM8K、BBH 等短任务分数相当[^src-moba-mixture-of-block-attention]。

### 效率

仅比较 attention 层前向：1M token prefill 加速至多 6.5 倍；固定块数与 top-k、按比例增大块大小以维持 95.31% 稀疏度（配合 query head 级张量并行），10M token 时相对 FlashAttention 加速 16 倍，计算复杂度次二次，加速来自块稀疏与 MoE/FlashAttention 联合实现[^src-moba-mixture-of-block-attention]。

## 范围与边界

Kimi 部署仅见于论文单一来源，无细节与线上指标[^src-moba-mixture-of-block-attention]。下游评测中 MoBA 仅用于 prefill，decode 用 full attention，论文未单独评估稀疏 attention 在生成期的质量[^src-moba-mixture-of-block-attention]。纯 MoBA 训练存在 trailing loss 缺口，需混合策略修复；混合配比（90/10 两阶段、末 3 层 full）来自其实验设置[^src-moba-mixture-of-block-attention]。MiTA 论文将块路由专家判为「仅位置驱动」的刚性划分，主张专家应随内容形变；该论断出自分析论证，未与 MoBA 同台评测[^src-2602-01219-mita]。选择粒度与 [[deepseek-sparse-attention|DSA]] 不同：MoBA 以块为单位（$N/n$ 个专家），DSA 以单个 MLA KV latent 为单位（$N$ 个宽 $k$ 的专家），代价是 DSA 的打分步骤仍为 $\mathcal{O}(L^2)$（§2.3）[^src-2512-02556-deepseek-v3-2]。Quest 可视为更小块加 min/max pooling 块表示的 MoBA，LongHeads 可视为 top-1 MoBA（出自该 related work）；H2O、StreamingLLM、TOVA、FastGen 属 decode 侧 KV 压缩，与 MoBA 的训练期架构稀疏正交（见 [[kv-cache-compression]]）[^src-moba-mixture-of-block-attention]。

## 与 NSA 的对照

两者都把 KV 切片后再选，支路数与合并方式不同：

MoBA 为单支路，门控在块集合上取 top-k，选中块与当前块直接做 attention，top-k 覆盖全部块时退化为 full attention[^src-moba-mixture-of-block-attention]；[[native-sparse-attention|NSA]] 采用压缩、块选择、滑动窗口三支路，各自 softmax 后按 sigmoid 门控加权，并以重映射后的 KV 替换原始 KV（式 3–5），无退化至 full attention 的开关（§3.2）[^src-2502-11089-nsa]。块打分上，MoBA 取 query 与块内 key 均值的内积，不引入新参数[^src-moba-mixture-of-block-attention]；NSA 直接从压缩支路的注意力分数取出块重要性分（式 8）[^src-2502-11089-nsa]。规模与阶段上，MoBA 缩放实验至 2.1B、下游到 Llama-8B 的 1M 上下文，评测期仅用于 prefill[^src-moba-mixture-of-block-attention]；NSA 在 27B 总参 / 3B 激活的模型上预训练，报告 prefill、decode、backward 三档加速[^src-2502-11089-nsa]。两者未做同台实验，NSA 的稀疏基线是 H2O、InfLLM、Quest、Exact-Top，质量基线是 Full Attention（§4.2）[^src-2502-11089-nsa]。

## 相关页面

[[mixture-of-experts]] 为 MoBA 直接借用的 top-k 门控原则，[[attention-sink]] 是 MoBA 归入静态稀疏特例的现象，[[kv-cache-compression]] 代表与 MoBA 训练期稀疏正交的推理期 KV 压缩路线，[[source-moba-mixture-of-block-attention]] 为源摘要。[[mita-attention]] 以形变 expert 替代定长块并采用 $N$-to-$m$ 路由，[[deformable-fast-weight-experts]] 对应刚性块与形变专家的对照，[[fast-weight-scaling-taxonomy]] 标出 MoBA 在五维分类中的格位。[[native-sparse-attention]] 对应三支路加门控合并的块稀疏对照，[[deepseek-sparse-attention]] 对应 token 粒度的 top-$k$ 对照，[[sdpa-output-gating]] 则是决定读取哪些 KV 的块级 top-$k$ 门控与输出侧逐元素门控的粒度对照，[[minimax-sparse-attention]] 同在 GQA 上的块稀疏对照：小块 + 专用索引头按 KL 对齐训练、逐组独立选块。

[^src-moba-mixture-of-block-attention]: [[source-moba-mixture-of-block-attention]]
[^src-2602-01219-mita]: [[source-2602-01219-mita]]
[^src-2512-02556-deepseek-v3-2]: [[source-2512-02556-deepseek-v3-2]]
[^src-2502-11089-nsa]: [[source-2502-11089-nsa]]
