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
last_updated: 2026-11-20
source_count: 1
confidence: medium
status: active
---

# Cross-Layer Attention（CLA）

只在模型的一个层子集中计算 K/V 投影，没有 K/V 投影的层复用前面层的 KV 激活；只有做投影的层往 KV cache 写数据（§2.2）[^src-2405-12981-cla]。一份 K/V 投影被多少层共用称为共享因子（sharing factor）$f$：CLA2 即相邻两层一对（$f=2$），CLA3 为三层一组，依此类推（§2.2）[^src-2405-12981-cla]。论文将其定位为 MQA/GQA 的直接延伸——把「query head 之间共享 K/V」推到「层之间共享 K/V」，在 MQA 之上再把 KV cache 缩 $f$ 倍（Abstract、§2.2）[^src-2405-12981-cla]。CLA 与 MHA/MQA/GQA 正交，可与其任意组合（§2.2）[^src-2405-12981-cla]。

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

**设置**（§3.1、Table 2）[^src-2405-12981-cla]：1B（hidden 2048、FFN 5472、20 层、30B token）与 3B（hidden 3072、FFN 8192、32 层、100B token）从零训练于 SlimPajama（GPT-NeoX tokenizer），Llama-like 架构（pre-norm、SwiGLU、RoPE、无 dropout），AdamW、batch $2048\times2048\approx 4$M token/step、seq 2048，H100 + PyTorch BF16。指标为 SlimPajama held-out $\approx$4M token 验证 ppl、Wikitext ppl 与 LM Eval Harness 七个下游任务（Hellaswag/PIQA/WG/SciQ/OBQA/BoolQ/ARC-E）。设计空间探索统一 LR $3\times10^{-4}$。

**Table 1**（1B 设计空间，LR $3\times10^{-4}$）[^src-2405-12981-cla]：

| 模型 | $d_{head}$ | Q heads | KV heads | KV layers | B/token | Val ppl |
|---|---|---|---|---|---|---|
| H128-MHA | 128 | 16 | 16 | 20 | 163 840 | 13.15 |
| H128-GQA4 | 128 | 16 | 4 | 20 | 40 960 | 13.36 |
| H128-GQA2 | 128 | 16 | 2 | 20 | 20 480 | 13.52 |
| H128-MQA | 128 | 16 | 1 | 20 | 10 240 | 13.54 |
| H64-MQA | 64 | 32 | 1 | 20 | 5120 | 13.81 |
| H46-MQA | 46 | 45 | 1 | 20 | 3680 | 13.96 |
| H32-MQA | 32 | 64 | 1 | 20 | 2560 | 14.37 |
| H512-MQA-CLA2 | 512 | 4 | 1 | 10 | 20 480 | 13.49 |
| H256-MQA-CLA2 | 256 | 8 | 1 | 10 | 10 240 | 13.51 |
| H128-MQA-CLA2 | 128 | 16 | 1 | 10 | 5120 | 13.60 |
| H90-MQA-CLA2 | 90 | 22 | 1 | 10 | 3600 | 13.73 |
| H64-MQA-CLA2 | 64 | 32 | 1 | 10 | 2560 | 13.89 |
| H256-GQA4-CLA2 | 256 | 8 | 4 | 10 | 40 960 | 13.38 |
| H128-GQA4-CLA2 | 128 | 16 | 4 | 10 | 20 480 | 13.48 |
| H128-GQA2-CLA2 | 128 | 16 | 2 | 10 | 10 240 | 13.59 |
| H128-MQA-CLA3 | 128 | 16 | 1 | 7 | 3584 | 13.77 |
| H128-MQA-CLA4 | 128 | 16 | 1 | 5 | 2560 | 13.95 |
| H128-MQA-CLA2-KeepEnds | 128 | 16 | 1 | 11 | 5632 | 13.62 |
| H128-MQA-CLA2-DenseFront | 128 | 16 | 1 | 11 | 5632 | 13.75 |
| H128-MQA-CLA2-DenseBack | 128 | 16 | 1 | 11 | 5632 | 14.03 |

读法（§3.2.1）[^src-2405-12981-cla]：非 CLA 基线 ppl 随 KV cache 容量单调恶化（13.15→14.37）；$d_{head}\in\{64,90,128\}$ 的 MQA-CLA2 对齐 $\{32,46,64\}$ 的 MQA 内存，ppl 改善 0.21–0.48 点（如 13.60 vs 13.81、13.89 vs 14.37）；$d_{head}\in\{256,512\}$ 的 CLA2 对齐 $d_{head}=128$ 的 MQA/GQA2 内存，仍改善 0.03 点——省下的层预算换成更大 head 维度是正收益。Figure 3 为该表的 Pareto 前沿绘图（红点 CLA 整体位于蓝点左下）。

**学习率调优**（§3.2.2、Table 3/4；LR 从 $3\times10^{-4}$ 起按 1.5× 递增扫描）[^src-2405-12981-cla]：

| 模型 | B/token | 最优 LR | Val ppl | Wikitext ppl |
|---|---|---|---|---|
| H128-MQA | 10 240 | $1.5\times10^{-3}$ | 12.39 | 19.30 |
| H128-MQA-CLA2 | 5120 | $2.25\times10^{-3}$ | 12.43 | 19.29 |
| H64-MQA | 5120 | $2.25\times10^{-3}$ | 12.74 | 20.00 |

调优后定性结论不变：CLA2 以 2× 更小缓存对同 $d_{head}$ 基线仅退 0.04 点，对同内存基线改善 0.31 点（Wikitext 上分别为 −0.01 与 +0.71）。CLA 模型的最优学习率更高（论文称 preliminary evidence，§3）。1B 下游（Table 4）三模型互有胜负、相差 1–5 个百分点；CLA2 相对 H128-MQA 在 7 项中输 5 项（BoolQ −4.19 最大），仅 OBQA +2.4、PIQA 平。

**3B 第一组**（§3.3、Table 5/6，LR 各自调优）[^src-2405-12981-cla]：

| 模型 | B/token | 最优 LR | Val ppl | Wikitext ppl |
|---|---|---|---|---|
| H128-MQA | 16 384 | $6.75\times10^{-4}$ | 9.52 | 13.63 |
| H128-MQA-CLA2 | 8192 | $2.25\times10^{-3}$ | 9.34 | 13.25 |
| H64-MQA | 8192 | $1.0\times10^{-3}$ | 9.48 | 13.49 |

与 1B 预期不同：CLA2 以一半缓存同时低于两个基线（val 9.34 对 9.52/9.48），且 H64-MQA 反超 H128-MQA（论文如实报告该异常）。下游（Table 6）CLA2 在 7 项中赢 4 项（Hellaswag 47.12、PIQA 74.32、WG 60.69、SciQ 89.2）。

**3B 第二组**（§3.3、Table 7/8；换训练集群、软件栈与数据顺序，统一 LR $10^{-3}$，基线重训以对齐环境）[^src-2405-12981-cla]：H64-MQA 12.94 / H64-MQA-CLA2 12.99（4096 B，−0.05 点）/ H32-MQA 13.34——同内存 +0.35 点、同 head 维度 −0.05 点，Wikitext 口径与 1B 一致；下游三模型仍无一致胜负（WG 掉 3.0 点为最大单项差异）。

**配置结论**（§4）[^src-2405-12981-cla]：测试范围内 MQA-CLA2 在给定内存预算与模型规模下持续取得最低验证 ppl（0.01 点以内）；$f>2$（CLA3 起）仍 Pareto 优于纯 MQA 但同内存输给 CLA2——加深共享不如维持 $f=2$ 并调 head 维度；三个非均匀模式全部更差且更费内存，首尾层特殊待遇无收益；对 $d_{head}$ 为 64/128 的 MQA，CLA2 以低于 1% 的 ppl 变化换 2× 缩减，论文将其推荐为对现有 MQA 架构的低风险保守改动。

## 边界

端到端推理效率（吞吐、延迟、实际可达 batch/序列长度）留作未来工作，全文无服务实测——"enabling inference with longer sequence lengths and larger batch sizes" 是容量推算而非测量（§4、Abstract）[^src-2405-12981-cla]。全部结论来自从零预训练，没有从已有 MQA checkpoint 改造 CLA 的实验（GQA 的 uptraining 转换在此无对应物；§3）[^src-2405-12981-cla]。规模止于 3B / 100B token；单 seed、1B 固定数据顺序、等 FLOPs 为近似口径；3B 第二组已出现环境漂移量级（0.05 ppl）与部分结论数字同量级。下游七任务分辨率弱（差异 0–5 点、无一致胜负），正当性主要依赖 ppl 与内存两轴。与量化、驱逐的正交性只是论证层面，无 CLA + KVQuant / CLA + H2O 组合实验（§1、§5.1）[^src-2405-12981-cla]。未说明下游评测的 few-shot 设置。层数加深（1B 20 层 / 3B 32 层之外）时收益如何变化无受控消融，只能由 cache 份数 $\lceil L/f\rceil$ 随 $L$ 线性缩减的结构性质外推 [INFERENCE]。

## 相关页面

[[source-2405-12981-cla]] 为源摘要；[[multi-query-attention]] 与 [[grouped-query-attention]] 压 head 维副本数，CLA 压层维副本数，两轴正交可组合（论文实测组合中 MQA+CLA2 最稳，GQA+CLA2 中仅 GQA2-CLA2 打平同内存 MQA-CLA2）；[[yoco-architecture]] 同走层共享轴但一步到位——单份全局 cache 消去层数因子，代价是换成 self/cross decoder-decoder 架构，CLA 保留标准 block 只删部分 KV 投影、收益 $f$；[[multi-head-latent-attention]] 压每 token 缓存字节（低秩 latent），层数因子仍在；[[kv-cache-compression]] 汇总各轴；[[paged-attention]] 管放置与共享，可与 CLA 叠加。

[^src-2405-12981-cla]: [[source-2405-12981-cla]]
