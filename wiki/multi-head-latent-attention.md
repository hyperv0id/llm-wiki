---
title: "Multi-head Latent Attention（MLA）"
type: technique
tags:
  - attention
  - kv-cache
  - efficient-inference
  - low-rank
  - llm
  - decoupled-rope
created: 2026-10-26
last_updated: 2026-10-27
source_count: 2
confidence: medium
status: active
---

# Multi-head Latent Attention（MLA）

MLA 把每个 token 的 key 与 value 联合下投影成低维 latent $c^{KV}_t$，推理时只缓存该 latent，用上投影在注意力内部恢复多头 K/V；由于上投影可被矩阵结合律折进 query 侧与输出侧投影（$W^{UQ}$、$W^O$），解码时无需物化 K/V（§2.1.2、附录 C）[^src-deepseek-v2-2405-04434]。它压缩的是「每 token 缓存什么」：不改头数（区别于 [[multi-query-attention|MQA]] 与 [[grouped-query-attention|GQA]] 的头/组共享）、不淘汰已缓存条目（区别于 [[kv-cache-compression|KV 淘汰与量化]]）、不减少被 attend 的 token 数（区别于 [[deepseek-sparse-attention|DSA]]）。

## 问题定位

MHA 每个 token 需缓存 $2n_hd_hl$ 个元素，$n_h$ 个头各存一份 K/V 构成推理期的显存与带宽瓶颈，直接限制 batch size 与序列长度（§2.1.1）[^src-deepseek-v2-2405-04434]。[[multi-query-attention|MQA]] 把所有头压到一份 K/V、[[grouped-query-attention|GQA]] 压到 $n_g$ 组，均以质量换 cache：论文在 7B dense 上做同参数对照，MHA 在四个难基准上明显优于 GQA(8 组) 与 MQA（表 8，见下「证据」）[^src-deepseek-v2-2405-04434]。MLA 的目标是兼得 MHA 的表示能力与比 GQA 更小的 cache。

## 机制

三处改动叠加（§2.1.2–2.1.3）[^src-deepseek-v2-2405-04434]：

KV 联合低秩压缩以 $c^{KV}_t=W^{DKV}h_t\in\mathbb{R}^{d_c}$ 作为 K 与 V 共用的压缩表示，$d_c\ll d_hn_h$，推理只缓存该表示，cache 从 $2n_hd_hl$ 降到 $d_cl$。query 低秩压缩将 $c^Q_t=W^{DQ}h_t\in\mathbb{R}^{d'_c}$ 经 $W^{UQ}$ 展开，它不减小 cache，用于降低训练期激活显存（"to reduce the activation memory during training"）。decoupled RoPE 则不将位置信息写入 latent，改用 $d^R_h$ 维 query 分支 $q^R_t=\operatorname{RoPE}(W^{QR}c^Q_t)$ 与所有头共享的 key $k^R_t=\operatorname{RoPE}(W^{KR}h_t)$ 承载旋转位置，内容分支走 latent，二者打分前拼接，cache 因此是 $(d_c+d^R_h)l$ 而非 $d_cl$。

论文采用 Decoupled RoPE 的原因在于：若把 RoPE 直接施加在 $k^C_t$ 上，$W^{UK}$ 会与位于 $W^{UQ}$ 与 $W^{UK}$ 之间且逐对位置变化的旋转矩阵耦合；因矩阵乘法不可交换，上投影无法被吸收，解码时必须为全部前序 token 重算 key（§2.1.3）[^src-deepseek-v2-2405-04434]。

## 数据流与维度

$h_t\in\mathbb{R}^{d}$ 为第 $t$ 个 token 的隐状态（DeepSeek-V2 取 $d=5120$），$n_h=128$ 为头数，$d_h=128$ 为单头内容维，$d_c=512$ 与 $d'_c=1536$ 分别为 KV 和 query 的 latent 维，$d^R_h=64$ 为解耦位置分支的单头维，$l=60$ 为层数。论文按单 token 书写，再显式把 $q_t,k_t,v_t$ 切成 $n_h$ 头（式 4–6）；切分在实现中即为 reshape，$W^O$ 的分头列切片为本页补写 [INFERENCE]。

### Stage 1 · 下投影：压出 latent 与位置 key

三条支路并行读同一个 $h_t$，互不依赖：

$$\begin{aligned}
c^{KV}_t &= W^{DKV}h_t && \text{// } [1, \bcancel{5120}]\times[\bcancel{5120}, 512]\to[1, 512]\ \text{——每个 token 一个 512 维 KV latent，16384 维的 } k/v \text{ 被压到 1/32，它是推理期唯一进 cache 的内容}\\
c^{Q}_t &= W^{DQ}h_t && \text{// } [1, \bcancel{5120}]\times[\bcancel{5120}, 1536]\to[1, 1536]\ \text{——query latent，只为降训练激活显存，不进 cache}\\
k^{R}_t &= \operatorname{RoPE}(W^{KR}h_t) && \text{// } [1, \bcancel{5120}]\times[\bcancel{5120}, 64]\to[1, 64]\ \text{——位置分支的共享 key，直接从 } h_t \text{ 产生、绕开 latent，因此它的旋转不污染可吸收路径}
\end{aligned}$$

### Stage 2 · 上投影：每头物化 K/V（朴素形态，训练与未吸收路径）

$$\begin{aligned}
k^{C}_t &= W^{UK}c^{KV}_t && \text{// } [1, \bcancel{512}]\times[\bcancel{512}, 16384]\to[n_h, 128]\ \text{——一个 latent 展开成 } 128 \text{ 个头各 128 维的内容 key}\\
v^{C}_t &= W^{UV}c^{KV}_t && \text{// } [1, \bcancel{512}]\times[\bcancel{512}, 16384]\to[n_h, 128]\ \text{——value 与 key 共用同一个 latent，这是「联合」压缩的含义}\\
q^{C}_t &= W^{UQ}c^{Q}_t && \text{// } [1, \bcancel{1536}]\times[\bcancel{1536}, 16384]\to[n_h, 128]\ \text{——query latent 展开成每头 128 维内容 query}\\
q^{R}_t &= \operatorname{RoPE}(W^{QR}c^{Q}_t) && \text{// } [1, \bcancel{1536}]\times[\bcancel{1536}, 8192]\to[n_h, 64]\ \text{——每头各 64 维位置 query，RoPE 在此之后施加}
\end{aligned}$$

### Stage 3 · 拼接内容与位置，形成每头的完整 Q/K

$$\begin{aligned}
q_{t,i} &= [\,q^{C}_{t,i};\ q^{R}_{t,i}\,] && \text{// } [128]\oplus[64]\to[192]\ \text{——第 } i \text{ 头用自己的内容块拼自己的位置块}\\
k_{j,i} &= [\,k^{C}_{j,i};\ k^{R}_{j}\,] && \text{// } [128]\oplus[64]\to[192]\ \text{——位置块 64 维不分头，同一份被 } n_h \text{ 个头复用}
\end{aligned}$$

### Stage 4 · 打分、归一、按头聚合、输出投影

$$\begin{aligned}
S_{tji} &= q^{\top}_{t,i}k_{j,i}\big/\sqrt{d_h+d^{R}_h} && \text{// } [n_h, \bcancel{192}]\times[\bcancel{192}, t]\to[n_h, t]\ \text{——消去 192 维，每头对全部 } t \text{ 个前序 token 一行打分，分母是 } \sqrt{192}\\
o_{t,i} &= \sum_{j\leq t}\operatorname{softmax}_j\!\left(S_{tji}\right)v^{C}_{j,i} && \text{// } [n_h, t]\times[t, \bcancel{128}]\to[n_h, \bcancel{128}]\ \text{——沿 } j \text{ 归一并加权，value 只取内容块，没有位置维}\\
u_t &= W^{O}[\,o_{t,1};\dots;o_{t,n_h}\,] && \text{// } [1, \bcancel{16384}]\times[\bcancel{16384}, 5120]\to[1, 5120]\ \text{——} n_h \text{ 个头重新合并回模型维}
\end{aligned}$$

### Stage 5 · 推理形态：上投影吸收进两侧，cache 只留 latent

cache 内容从「每 token 每层 16384 维 K 加 16384 维 V」变成「512 维 latent 加 64 维共享位置 key」。折叠矩阵离线算一次：

$$\begin{aligned}
\tilde{W}^{UQ}_i &= W^{UK\top}_iW^{UQ}_i && \text{// } [512, \bcancel{128}]\times[\bcancel{128}, 1536]\to[512, 1536]\ \text{——每头一张 latent 空间内的 query 矩阵：} W^{UK}_i \text{、} W^{UQ}_i \text{ 各取第 } i \text{ 头的 } 128 \text{ 行切片，512 行取代 16384 行}\\
\tilde{q}_{t,i} &= [\,\tilde{W}^{UQ}_i c^{Q}_t;\ q^{R}_{t,i}\,] && \text{// } [512, \bcancel{1536}]\times[\bcancel{1536}, 1]\to[512]\ \text{再}\oplus[64]\to[576]\ \text{——内容块被折到 latent 维，不再展开成多头 K 空间；位置块 } q^R \text{ 原样拼入}\\
S_{tji} &= \tilde{q}^{\top}_{t,i}[\,c^{KV}_j;\ k^{R}_j\,]\big/\sqrt{d_h+d^{R}_h} && \text{// } [1, \bcancel{576}]\times[\bcancel{576}, t]\to[1, t]\ \text{——直接对 cache 里的 latent 打分；内积维变 576，分母仍是训练时的 } \sqrt{192}\\
\hat{o}_{t,i} &= \sum_{j}\operatorname{softmax}_j(S_{tji})\,c^{KV}_j,\quad u_t=\sum_i\left(W^{O}_iW^{UV}_i\right)\hat{o}_{t,i} && \text{// } [5120, \bcancel{512}]\times[\bcancel{512}, 1]\to[5120]\ \text{——加权和在 latent 空间完成，最后由折叠后的 } W^O_iW^{UV}_i \text{ 抬回模型维}
\end{aligned}$$

$W^O_i$ 是 $W^O\in\mathbb{R}^{5120\times16384}$ 中属于第 $i$ 头的 128 列切片，故 $W^O_iW^{UV}_i\in\mathbb{R}^{5120\times512}$。

## KV cache 逐项对比

每 token 缓存元素数（$l$ 层合计），公式取自论文表 1，最后一列是代入 DeepSeek-V2 配置的复算[^src-deepseek-v2-2405-04434]：

| 机制 | 论文公式 | 代入 $n_h=128$, $d_h=128$, $d_c=512$, $d^R_h=64$, $l=60$ |
|---|---|---|
| MHA | $2n_hd_hl$ | $1{,}966{,}080$ |
| GQA | $2n_gd_hl$ | 与 $n_g=2.25$ 等值 |
| MQA | $2d_hl$ | $15{,}360$ |
| MLA | $(d_c+d^R_h)l$，论文记作 $\approx\frac{9}{2}d_hl$ | $34{,}560$ |

代入链条[^src-deepseek-v2-2405-04434]：

由 $d_c=4d_h=512$、$d^R_h=d_h/2=64$ 得 $(d_c+d^R_h)l=4.5d_hl=\frac{9}{2}d_hl$；与 GQA 的 $2n_gd_hl$ 相等即 $n_g=2.25$，表 1 脚注据此得出「KV cache 等于 2.25 组的 GQA」。相比同配置 MHA，$34{,}560/1{,}966{,}080=1.76\%$ 即 K 与 V 合计缩小 56.9 倍；只看 K 侧 $16384\to576$ 为 28.4 倍，多出的 64 维即 decoupled RoPE 的代价（相对 $d_cl$ 增加 12.5%）。按 FP16 存储，每 token 的 latent 与位置 key 在 60 层合计 $576\times60\times2=69{,}120$ 字节（67.5 KiB），同配置 MHA 为 $1{,}966{,}080\times2=3.75$ MiB。

摘要中「KV cache 降低 93.3%」并非针对同配置 MHA，其口径是部署中的 DeepSeek 67B（[[grouped-query-attention|GQA]]）叠加 MLA 侧 KV cache 量化到平均 6 bit（§3.2.3）。按 67B 的 8 组 GQA、95 层、$d_h=128$、16 bit 缓存复算 [INFERENCE]：

$$1-\frac{(d_c+d^R_h)l\cdot 6}{2n_gd_hl_{67\mathrm{B}}\cdot 16}=1-\frac{34{,}560\times6}{194{,}560\times16}=1-0.0666\approx 93.3\%$$

其中 67B 的层数与组数取自 DeepSeek 67B 公开配置，论文未列这组参数；若换成 16 bit 的 MLA，比值是 17.8%，得不到 93.3%。

## 吸收上投影为何不改变数学结果

打分侧等价是矩阵结合律的直接结果，逐头成立（附录 C 原文 "due to the associative law of matrix multiplication"）[^src-deepseek-v2-2405-04434]：

$$\left(W^{UQ}_ic^{Q}_t\right)^{\!\top}\left(W^{UK}_ic^{KV}_j\right)=c^{Q\top}_t\left(W^{UQ\top}_iW^{UK}_i\right)c^{KV}_j=\tilde{q}^{\top}_{t,i}\,c^{KV}_j$$

右侧将「先上投影 K 再内积」改为「先在 latent 空间投影 query 再内积」，消去的 $d_hn_h=16384$ 维不再出现。

value 侧等价多一步，因 softmax 权重仅依赖 $q,k$、与 $v$ 无关：

$$u_t=\sum_iW^{O}_i\Big(\sum_ja_{tji}W^{UV}_ic^{KV}_j\Big)=\sum_i\left(W^{O}_iW^{UV}_i\right)\Big(\sum_ja_{tji}c^{KV}_j\Big)$$

即先在 512 维 latent 上加权求和，再用折叠后的 $5120\times512$ 矩阵抬回模型维。两式的共同前提是：内容路径上不存在夹在 $W^{UQ}$ 与 $W^{UK}$ 之间、逐位置对变化的矩阵——这正是 Stage 1 将 RoPE 拆至独立分支的原因[^src-deepseek-v2-2405-04434]。

计算量呈「乘加升、访存降」。论文仅声明省去重算 K/V 开销，未给 FLOPs 数字，以下按本页配置复算 [INFERENCE]：

query 上投影从 $n_hd_hd'_c=25.17$M 乘加增至吸收后 $n_hd_cd'_c=100.66$M（4 倍），源于每头都要在 512 维 latent 上单独出一份 query；每个 (头, 前序 token) 对的打分由 192 维内积变为 576 维内积（3 倍）；value 侧 $W^{UV}$ 的 8.39M 加 $W^O$ 的 83.89M 在吸收后变为 $n_h\,d\,d_c=335.54$M（3.6 倍）；相应地，每 token 每层从 HBM 读取的 K/V 由 $2\times16384$ 个元素降至 576 个（56.9 倍）。

论文省去朴素形态下每步为全部前序 token 重算 $W^{UK}$ 与 $W^{UV}$ 的乘加（每层每 token $d_hn_hd_c=8.39$M）；吸收后该项从解码循环中消失[^src-deepseek-v2-2405-04434]。

逐步解码受限于带宽，访存降幅远大于算力增幅，因而采用吸收形式后吞吐上升；训练仍需物化 K/V 做反向传播，吸收仅在推理成立。

## 证据

**同尺寸对照（dense 7B，1.33T token，调层数对齐参数量）**：MHA 在 BBH/MMLU/C-Eval/CMMLU 上为 37.0/45.2/42.9/43.5，GQA(8 组) 35.6/41.2/37.7/38.4，MQA 33.2/37.9/30.0/34.6（表 8，§D.1）[^src-deepseek-v2-2405-04434]。数据表明，削减 cache 组数或头数在难基准上有明确代价，这也是 MLA 的设计动机。

**MLA vs MHA（MoE，两个规模，除注意力外架构相同）**（表 9，§D.2）[^src-deepseek-v2-2405-04434]：

| 设置 | 总/激活参数 | 每 token KV cache | BBH | MMLU | C-Eval | CMMLU |
|---|---|---|---|---|---|---|
| 小 MoE + MHA | 15.8B / 2.5B | 110.6K | 37.9 | 48.7 | 51.6 | 52.3 |
| 小 MoE + MLA | 15.7B / 2.4B | 15.6K | 39.0 | 50.0 | 50.9 | 53.4 |
| 大 MoE + MHA | 250.8B / 25.0B | 860.2K | 46.6 | 57.5 | 57.9 | 60.7 |
| 大 MoE + MLA | 247.4B / 21.5B | 34.6K | 50.7 | 59.0 | 59.2 | 62.5 |

两个规模分别训练 1.33T 与 420B token。MLA 表现优于 MHA，且 cache 仅为 MHA 的 14%（小）与 4%（大）；C-Eval 小模型上 MLA 低 0.7，是两规模 × 四基准共八项中唯一落后的一格[^src-deepseek-v2-2405-04434]。论文未提供两者的训练损失曲线，质量对照仅有这四个基准的分数（图 1 仅作 MMLU、激活参数与吞吐的外部对比，不含损失）。MLA 侧 $34.6\text{K}=(512+64)\times60$ 可精确复算；MHA 侧 $860.2\text{K}$ 与正文 $n_h=128$、$d_h=128$、$l=60$ 给出的 $1966\text{K}$ 不自洽，反推每层 $n_hd_h=7168$，表明该对照的 MHA 模型采用了其他头配置。

**吞吐与成本（§3.2.3）**：单节点 8 张 H800 上，DeepSeek-V2 生成吞吐 >50K token/s，达到 DeepSeek 67B 最高吞吐的 5.76 倍；prompt 输入吞吐 >100K token/s。该数据基于 FP8 权重与平均 6 bit KV cache 量化，对照 dense 67B，包含 MoE 少激活参数、FP8 及 MLA 的共同贡献。训练侧每万亿 token 由 67B 的 300.6K GPU 小时降至 172.8K（省 42.5%），同样由 MoE 与 MLA 共同达成[^src-deepseek-v2-2405-04434]。

## 边界

压缩依赖 $d_c$ 足够大：$d_c=512$ 是 $d_hn_h=16384$ 的 1/32，仍是 MQA 单份 K/V（128 维）的 4 倍；latent 需同时承载 K 与 V 的全部头，$d_c$ 过小时两个上投影互相挤占子空间。论文未给出 $d_c$ 的扫描或消融（§D 仅有 MHA/GQA/MQA 与 MLA-vs-MHA 两组对照），$d_c=4d_h$ 属超参选择而非实验结论[^src-deepseek-v2-2405-04434]。decoupled RoPE 是位置与内容解耦的代价而非压缩项，使每层每 token 多缓存 64 个元素（相对 $d_cl$ 增加 12.5%），以换取上投影被吸收；若去除，$W^{UK}$ 无法折进 $W^{UQ}$，解码须重算全部前序 key。训练侧显存并不因 KV cache 降低，因其仅在推理期存在，训练须物化 K/V 与反向中间量，MLA 在训练期节省的是 query 侧激活（$W^{DQ}\to W^{UQ}$ 的低秩瓶颈）。实验证据来自单一技术报告的单次无方差训练，质量对照仅覆盖四个多选题难基准，长上下文与生成质量未与 GQA/MQA 同台比较；大 MoE 只训练 420B token，少于小 MoE 的 1.33T。模型头维与层数按 DeepSeek-V2 主模型取值，DeepSeek-V2-Lite 为 27 层、16 头、$d_h=128$（表 9 小模型 $15.6\text{K}=576\times27$）[^src-deepseek-v2-2405-04434]。

## 相关页面

[[multi-query-attention]] 削减头数，所有 query 头共享一份 K/V，cache 最小但质量损失最大（表 8 MQA 33.2 BBH）；[[grouped-query-attention]] 削减组数，组内共享 K/V，介于 MHA 与 MQA 之间，MLA 在 cache 上等价于 2.25 组且质量不低于 MHA；[[kv-cache-compression]] 减少已缓存条目，在推理期淘汰或量化 token 与 bit 数，与模型设计正交，V2 部署即叠加 6 bit 量化；[[deepseek-sparse-attention]] 缩减被 attend 的 token 数，在 MLA latent 上通过 top-$k$ 选择降低 $\mathcal{O}(L^2)$ 计算，MLA 本身不减 token 数；[[rope]] 为 decoupled RoPE 规避的旋转矩阵不可交换性；[[deepseek-v3-2]] 与 [[source-2512-02556-deepseek-v3-2]] 均沿用 MLA。

MLA 之外的一种用法：Kimi Linear（Moonshot AI）在 3:1 KDA/MLA 层间混合中把 Full MLA 用作全局注意力支路并对全部 MLA 层用 NoPE——位置信息交给 [[kimi-delta-attention|KDA]] 的门控衰减承载，免掉 RoPE base 调整，且 NoPE 的 MLA 可在推理期转成纯 MQA（§4）[^src-moonshot-2025-kimi-linear]。

[^src-deepseek-v2-2405-04434]: [[source-deepseek-v2-2405-04434]]
[^src-moonshot-2025-kimi-linear]: [[source-moonshot-2025-kimi-linear]]
