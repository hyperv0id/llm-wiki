---
title: "Cross-Layer Attention (CLA)"
type: technique
tags:
  - attention
  - kv-cache
  - efficient-inference
  - llm
  - transformer
created: 2026-11-20
last_updated: 2026-11-22
source_count: 1
confidence: medium
status: active
---

# Cross-Layer Attention（CLA）

CLA 只在一部分层计算 K/V 投影，其余层直接复用所属组第一层的 KV 激活，因此只有做投影的层往 KV cache 写数据。一份 K/V 投影被多少层共用是共享因子（sharing factor）$f$：CLA2 为相邻两层一对，CLA3 为三层一组，依此类推。论文把它定位为 MQA/GQA 的直接延伸——从「query head 之间共享 K/V」推到「层之间共享 K/V」，在 MQA 之上再把 cache 缩 $f$ 倍，且与 MHA/MQA/GQA 正交、可任意组合（Abstract、§2.2）[^src-2405-12981-cla]。

## 问题定位

KV cache 大小与序列长度、batch size 同时成正比：长序列下限制 batch size，显存紧张时被迫 offloading，KV cache 的长期持久化（避免重复计算）成本也直接由 cache 大小决定（§1）[^src-2405-12981-cla]。论文把既有干预归为低精度存储、KV 条目驱逐、query head 间共享（MQA/GQA）三类，CLA 沿第四条轴——减少 KV cache 中不同层的数量（§1）[^src-2405-12981-cla]。目标只是推理期 KV cache 的存储占用：单次前向内的带宽（FlashAttention）与单次前向内的存储（FlexGen offloading）不在范围内（§5）[^src-2405-12981-cla]。

## 结构

**block 构成**（Figure 1）[^src-2405-12981-cla]：有 K/V 投影的层与标准 Transformer block 完全一致；复用层把 block 内的 K/V 投影整块删除、只保留 Q 投影，注意力读同组共享的 cache，FFN、输出投影、Norm、残差逐层独立。CLA 模型对 attention 中的 K/V 投影块与 Q 投影块使用各自独立可学习的 affine LayerNorm 参数（§3.1）[^src-2405-12981-cla]。

**分组规则**（Figure 2）[^src-2405-12981-cla]：均匀划分、从第 0 层起每 $f$ 层一组，组内第一层承担 K/V 投影。$f$ 不整除层数时某组层数少于 $f$，论文把短组放在最前端（layer 0 的 cache 只服务 layer 0，措辞为 "arbitrarily select"）；20 层上 CLA2 得 10 份 cache、CLA3 得 $\lceil 20/3\rceil=7$ 份、CLA4 得 5 份（Figure 2、Table 1）[^src-2405-12981-cla]。

**非均匀变体**（§3.2.1）[^src-2405-12981-cla]：KeepEnds（layer 0 与最后一层各独占一份 cache）、DenseFront（10 个非 CLA 层 + 9 层共用 layer 9 的 KV）、DenseBack（2 个非 CLA 层 + 10 层共用 layer 1 的 KV + 9 个非 CLA 层）。

## 数据流与维度变换

**维度符号**：$L$ 总层数；$f$ 共享因子；$n_{query}$、$n_{kv}$ 为 query 与 K/V 头数（MQA 时 $n_{kv}=1$，MHA 时 $n_{kv}=n_{query}$）；$d_{head}$ 单头维度，约定 $n_{query}\, d_{head}=d_{model}$；$N$ 单次前向 query 位置数；$M$ 被注意的已缓存位置数；$a(l)=f\lfloor l/f\rfloor$ 为层 $l$ 所在组的组首层。形状记 $[\text{行},\text{列}]$。

### Stage 1 · 组首层产出共享 K/V

仅 $l\in\{0,f,2f,\dots\}$ 的层执行（$f=2$ 时即偶数层）：

$$\begin{aligned}
K^{a(l)} &= \operatorname{LN}_K(X^{a(l)})\,W_K^{(a(l)}) && \text{// } [M, \bcancel{d_{model}}]\times[\bcancel{d_{model}}, n_{kv} d_{head}]\to[M, n_{kv} d_{head}]\ \text{——写入 cache 的唯一来源，组内 } f \text{ 层共用}\\
V^{a(l)} &= \operatorname{LN}_K(X^{a(l)})\,W_V^{(a(l)}) && \text{// } [M, \bcancel{d_{model}}]\times[\bcancel{d_{model}}, n_{kv} d_{head}]\to[M, n_{kv} d_{head}]\ \text{——与 } K \text{ 同源同时算出，此后对组内其余层冻结}
\end{aligned}$$

### Stage 2 · 复用层只投 Q

$$\begin{aligned}
Q^{l} &= \operatorname{LN}_Q(X^{l})\,W_Q^{(l)} && \text{// } [N, \bcancel{d_{model}}]\times[\bcancel{d_{model}}, n_{query} d_{head}]\to[N, n_{query} d_{head}]\ \text{——复用层删除 } W_K, W_V\text{，} \operatorname{LN}_Q \text{ 与 } \operatorname{LN}_K \text{ 参数独立}
\end{aligned}$$

### Stage 3 · 组内各层读同一份 cache 做标准注意力

$$\begin{aligned}
O^{l} &= \operatorname{Softmax}\!\left(Q^{l} (K^{a(l)})^{\top}\big/\sqrt{d_{head}}\right) V^{a(l)} && \text{// } [N, \bcancel{d_{head}}]\times[\bcancel{d_{head}}, M]\times[\bcancel{M}, d_{head}]\ \text{——每层 } Q \text{ 各异、} K/V \text{ 同份；注意力算术量与逐层 KV 的模型相同}
\end{aligned}$$

### Stage 4 · 解码增量：新 token 只在组首层追加

$$\begin{aligned}
K^{a(l)}_t &= [\,K^{a(l)}_{t-1};\ k_t\,] && \text{// } [M, n_{kv} d_{head}]\oplus[1, n_{kv} d_{head}]\to[M{+}1, n_{kv} d_{head}]\ \text{——单步只写 } \lceil L/f\rceil \text{ 份 cache，每层各自的注意力仍要重读它}
\end{aligned}$$

### Stage 5 · cache 记账：层数因子除以 $f$

$$\begin{aligned}
n_{\text{cache}} &= \lceil L/f\rceil && \text{// } L \text{ 份压到 } \lceil L/f\rceil \text{ 份——不整除时该值使实际缩比略小于 } f\\
B_{\text{CLA}} &= 4\,n_{kv}\, d_{head}\,\lceil L/f\rceil \ \text{bytes/token} && \text{// } K,V\text{ 两张量}\times 2\ \text{bytes（16-bit）}\times n_{kv} d_{head}\ \text{每份——1B 上 H128-MQA-CLA2：} 4\cdot 128\cdot 10=5120
\end{aligned}$$

相对 MHA 的总缩比为 $(n_{query}/n_{kv})\cdot f$；相对 MQA 为 $f$（Abstract 的 "another 2×"）。1B（20 层）上 H128-MHA 163840 → H128-MQA 10240 → H128-MQA-CLA2 5120 bytes/token，即 MQA 基础上再除 2、相对 MHA 共 32×（Table 1）[^src-2405-12981-cla]。

## 系统影响

**省容量，不省带宽**：与 MQA/GQA 不同，CLA 对每步解码中 attention 消耗的内存带宽没有直接影响——共享的 KV cache 在每个注意力层仍要分别从主存重读（§2.3）[^src-2405-12981-cla]。收益落在容量上限换来的更大 batch、更长序列与更长的 cache 持久化时间。**并行**：与 tensor parallelism 完全兼容；pipeline parallelism 下共享同一份 cache 的层必须在同一 stage，否则需跨 stage 传 KV（§2.3）[^src-2405-12981-cla]。**参数与 FLOPs**：删除部分 K/V 投影块使参数量与前向/反向 FLOPs 略降（§2.3）[^src-2405-12981-cla]——对照实验「同等 FLOPs」是近似口径（§3.2）[^src-2405-12981-cla]。**训练显存**：训练中物化的中间 KV 激活张量减少，但对 GQA/MQA 模型这类张量本就远小于 hidden states 与 MLP 激活（§2.3）[^src-2405-12981-cla]。

## 证据

**设置**：1B（hidden 2048、20 层）与 3B（hidden 3072、32 层）两组 Llama-like 模型从零预训练于 SlimPajama（1B 30B token、3B 100B token），指标是 held-out ppl 加 LM Eval Harness 七个下游任务；设计空间统一 LR $3\times10^{-4}$，最终结论按各自调优 LR 给出 [^src-2405-12981-cla]。

**定位与主结论**：不是精度超越，而是同等精度下把 KV cache 再减一半。1B 上 H128-MQA-CLA2 以 5120 bytes/token 对 H128-MQA 的 10240 取得 13.60 对 13.54，学习率调优后 12.43 对 12.39，Wikitext 上反而持平（19.29 对 19.30）；同内存的 H64-MQA（5120）被拉开 0.2 点（13.81），head 维度更小时差距到 0.48 点（13.89 对 14.37）。3B 第一组反过来：CLA2 以一半缓存同时低于 H128-MQA 与 H64-MQA（9.34 对 9.52 / 9.48），只是同组 H64-MQA 的参数更少却高于 H128-MQA（论文如实报告的异常），说明该规模下的差距已在训练噪声量级 [^src-2405-12981-cla]。

| 模型 | B/token | Val ppl（1B） |
|---|---:|---:|
| H128-MHA | 163 840 | 13.15 |
| H128-MQA | 10 240 | 13.54 |
| H128-MQA-CLA2 | 5120 | 13.60 |
| H128-MQA-CLA3 | 3584 | 13.77 |
| H128-MQA-CLA4 | 2560 | 13.95 |

**涨点归因**：收益来自「省下的层预算换更大的 head 维度」，不是加深共享——CLA2/CLA3/CLA4 依次 13.60/13.77/13.95，$f>2$ 仍 Pareto 优于纯 MQA 但同内存输给 CLA2；$d_{head}=128$ 时 CLA2 比同内存 MQA 好 0.2 点，head 更小时好 0.48 点，到 $d_{head}=256/512$ 对齐 $d_{head}=128$ 基线的内存时只剩 0.03 点。三个非均匀模式（KeepEnds 13.62、DenseFront 13.75、DenseBack 14.03）全部更差且多占内存，首尾层特殊待遇没有收益。CLA 模型的最优学习率也高于纯 MQA（$2.25\times10^{-3}$ 对 $1.5\times10^{-3}$），论文称其为 preliminary evidence [^src-2405-12981-cla]。

**证据缺口与负结果**：七个下游任务的差异只有 1–5 个百分点且无一致胜负——1B 上 CLA2 相对 H128-MQA 输 5 项、赢 1 项、平 1 项（BoolQ −4.19 为最大负项，仅 OBQA +2.4）；3B 第二组换训练集群与软件栈后，同 head 维度退 0.05 点，同内存优势缩到 0.35 点。全文没有服务实测：吞吐、延迟与实际可达的 batch/序列长度都是容量推算而非测量；也没有从已有 MQA checkpoint 改造 CLA 的实验；规模止于 3B / 100B token、单 seed，CLA 与量化、驱逐的组合只有论证没有实验 [^src-2405-12981-cla]。

## 边界

- 1B 固定数据顺序，等 FLOPs 对照只是近似口径（§3.2）；下游评测的 few-shot 设置未说明。
- 层数继续加深（超过 1B 的 20 层 / 3B 的 32 层）时收益如何变化没有受控消融，只能由 cache 份数 $\lceil L/f\rceil$ 随 $L$ 线性缩减的结构性质外推 [INFERENCE]。
- 论文把 CLA 推荐为对现有 MQA 架构的低风险改动，依据是 ppl 与内存两轴，下游任务不承担这个判断 [^src-2405-12981-cla]。

## 相关页面

[[source-2405-12981-cla]] 为源摘要；[[multi-query-attention]] 与 [[grouped-query-attention]] 压 head 维副本数，CLA 压层维副本数，两轴正交可组合（论文实测组合中 MQA+CLA2 最稳，GQA+CLA2 中仅 GQA2-CLA2 打平同内存 MQA-CLA2）；[[yoco-architecture]] 同走层共享轴但一步到位——单份全局 cache 消去层数因子，代价是换成 self/cross decoder-decoder 架构，CLA 保留标准 block 只删部分 KV 投影、收益 $f$；[[multi-head-latent-attention]] 压每 token 缓存字节（低秩 latent），层数因子仍在；[[kv-cache-compression]] 汇总各轴；[[paged-attention]] 管放置与共享，可与 CLA 叠加。

[^src-2405-12981-cla]: [[source-2405-12981-cla]]
