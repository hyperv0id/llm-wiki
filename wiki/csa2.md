---
title: "Compressed Sparse Attention 2 (CSA2)"
type: technique
tags:
  - attention
  - sparse-attention
  - long-context
  - kv-cache
  - llm
created: 2026-10-27
last_updated: 2026-10-27
source_count: 2
confidence: medium
status: active
---

# Compressed Sparse Attention 2 (CSA2)

CSA2 是 DeepSeek-V4.1-Flash 唯一的全局注意力形态，替换 V4 的 [[compressed-sparse-attention|CSA]]–[[heavily-compressed-attention|HCA]] 混合：主 KV 与 indexer K 沿**层维**跨层共享，Top-$k$ 下标亦跨层复用，压缩器与 indexer 同时简化（§2.3、§1）[^src-deepseek-v4-1-2609-19969]。论文将 KV 成本归纳为三个可乘维度——entry 大小（GQA/MLA）、序列维（每 $m$ 个 token 压成 1 条）、层维（复用别层的 cache 与选择），CSA2 同时用满三者（§2.3）[^src-deepseek-v4-1-2609-19969]。

## 三种静态模式

每层在训练前被静态指定为 Full、Reindex、Reuse 之一；三种模式都自算 global Q 与本层 SWA KV，差别只在主 KV、indexer K、Top-$k$ 下标从哪来（§2.3.1、Figure 4）[^src-deepseek-v4-1-2609-19969]：

**Full**：本层计算主 KV、投影出 indexer K 并运行 indexer 产生新 Top-$k$ 下标，是唯一完整执行 CSA2 计算的层；**Reindex**：复用最近一个 Full 层的主 KV 与 indexer K，用本层 indexer Q 重新打分产生新下标，共享 cache 但允许逐层改变选中集合；**Reuse**：复用最近的主 KV 与针对该 KV 计算出的最新下标，不计算 indexer Q 也不打分，直接做稀疏注意力。

与 CED 组合时，解码器中设为 Full 的层从第 $L/2$ 层隐藏状态计算自身全局 KV，Reindex/Reuse 保持不变（§2.3.1）[^src-deepseek-v4-1-2609-19969]。

## 相对 CSA 的逐项差异

CSA（V4）每 $m=4$ 个 token 压缩为 1 条 entry，每条由 $2m$ 个原始 entry 加权得到，相邻条目共享重叠的 $b$ 支路，压缩时加入可学习绝对位置偏置，indexer K 则采用从隐藏状态出发的独立压缩路径（§2.3 式 9–12）[^src-deepseek-v4-2606-19348]。CSA2 的改动按原文顺序排列（§2.3），表中配置与精度项见 §2.3.1、§2.4.4、§4.2.1[^src-deepseek-v4-1-2609-19969]：

| 项 | CSA（V4） | CSA2（V4.1） |
|---|---|---|
| 压缩单元 | 每 1 个条目由 $2m$ 个原始条目算出，$a/b$ 双支路跨块重叠 | 每 1 个条目由 $m$ 个原始条目算出，无重叠 |
| 位置信息 | 压缩时加可学习绝对位置偏置 $B^{a},B^{b}$ | 去掉绝对位置嵌入 |
| indexer K 来源 | 从隐藏状态另走一条压缩路径 | 由主 KV 条目投影 |
| 层间共享 | 无，每层各持一套主 KV | Full/Reindex/Reuse 三模式共享主 KV 与 indexer K，可复用下标 |
| 无压缩（$m=1$） | 无此形态；全局分支始终带压缩，长程覆盖由 HCA（$m'=128$、dense 不选）承担 | $m=1$ 是 CSA2 的特例，解码器全用它；无 HCA |
| 压缩率 | $m=4$（Flash 与 Pro 同） | 编码器 $m=2$，解码器 $m=1$ |
| indexer 头数 / 头维 | 64 / 128 | 32 / 128 |
| top-$k$ | 512（Flash）/ 1024（Pro） | 512 |
| query 压缩维 | 1024 / 1536 | 1280 |
| 每层打分范围 | 全部因果可见压缩条目 | 解码器首个 Full 层全长，后续 Reindex 层限于 16,384 位置候选池 |
| 主 KV 精度 | RoPE 维 BF16、其余 FP8 | FP4（E2M1，每 16 通道 1 个 E4M3 scale） |

压缩率从 $1/4$ 放宽至 $1/2$ 与 $1/1$ 却把 KV 压得更低，源于层维共享：解码器只有 1 个 Full 层物化全局 KV，其余 19 层复用（§2.3.1、§4.2.1）[^src-deepseek-v4-1-2609-19969]。原文仅描述压缩器差异而未重写公式，Stage 1–2 按 CSA 式 (11)(12) 去掉重叠与偏置后的同形写法给出，标注 [INFERENCE]。

## Hierarchical Sparse Indexer

跨层复用减少了 indexer **次数**，但剩余 indexer 仍须对全部因果可见位置打分，是长上下文下的主要瓶颈（§2.3.2）[^src-deepseek-v4-1-2609-19969]。解码器首个 Full 层在筛选自身 Top-$k$ 的同时做块级筛选：每 8 个位置为一块，块分取块内最大 index 分，取分最高的至多 2,048 块，覆盖位置构成候选池（例：$2{,}048\times8=16{,}384$ 个位置）；后续 Reindex 层仅在池内打分并选出自身 Top-512，Reuse 层不打分（§2.3.2、Figure 5、§4.2.1）[^src-deepseek-v4-1-2609-19969]。候选池大小固定后，深层 indexer 单 query 打分成本与上下文长度无关，仅首个 Full 层仍扫描全长（§2.3.2）[^src-deepseek-v4-1-2609-19969]。该方案为 training-aware，在 post-training 引入，训练与推理采用同一搜索域；预训练阶段 Reindex 层的打分范围原文未交代 [INFERENCE]。

## 数据流与维度变换

**维度符号**：$N$ 序列长度；$d=5120$ 隐藏维；$m$ 压缩率（编码器 2 / 解码器 1）；$c=512$ 主 KV 条目维兼单头 query 维；$n_h=64$ query 头数；$d_c=1280$ query 压缩维；$n^{I}_h=32$ indexer 头数；$c^{I}=128$ indexer 头维；$k=512$ top-$k$；候选池 $|\mathcal{P}|=16{,}384$；$n_{win}=128$ 滑窗条目；$g=8$ 输出分组数，$d'_g=1024$（§4.2.1）[^src-deepseek-v4-1-2609-19969]。形状记 $[\text{行},\text{列}]$；论文未给的实现细节标 [INFERENCE]。

### Stage 1 · Full 层算主 KV 与压缩权重

$$\begin{aligned}
C^{l} &= H^{l}W^{KV}_{l} && \text{// } [N,\bcancel{d}]\times[\bcancel{d},c]\to[N,c]\ \text{——每 token 一条未压缩 KV 条目，只有 Full 层算}\\
Z^{l} &= H^{l}W^{Z}_{l} && \text{// } [N,\bcancel{d}]\times[\bcancel{d},c]\to[N,c]\ \text{——逐 token 的压缩权重打分，与 } C \text{ 同形}
\end{aligned}$$

编码器 Full 层取本层隐藏状态，解码器 Full 层取 $H^{L/2}$（见 [[ced]]）；正文未重写 CSA 式 (9)(10)，此处按同形补齐 [INFERENCE]（§2.3）[^src-deepseek-v4-1-2609-19969]。

### Stage 2 · 非重叠分块压缩

$$\begin{aligned}
S^{l}_{i} &= \operatorname{Softmax}_{\text{row}}\!\left(Z^{l}_{mi:m(i+1)-1}\right) && \text{// } [m,c]\xrightarrow{\text{沿 } m \text{ 维逐通道归一}}[m,c]\ \text{——块内 } m \text{ 个 token 的权重，无偏置、无跨块项}\\
C^{Comp,l}_{i} &= \sum_{b=0}^{m-1}S^{l}_{i,b}\odot C^{l}_{mi+b} && \text{// } [m,c]\odot[m,c]\xrightarrow{\text{沿 } m \text{ 维逐通道加权求和}}[c]\ \text{——每个通道各自做 } m \text{ 项凸组合}\\
C^{Comp,l} &= \left\{C^{Comp,l}_{i}\right\}_{i=0}^{N/m-1} && \text{// } [N,c]\xrightarrow{\text{按 } m \text{ 非重叠分块}}[N/m,c]\ \text{——相邻条目不再共享 token}
\end{aligned}$$

$m=1$ 时该式退化为恒等，解码器即「未压缩主 KV」（§2.3）[^src-deepseek-v4-1-2609-19969]。正文未给 softmax 归一轴与权重参数化，按 CSA 口径补 [INFERENCE]。

### Stage 3 · 由主 KV 投影出 indexer K

$$\begin{aligned}
K^{I,Comp,l} &= C^{Comp,l}W^{IK}_{l} && \text{// } [N/m,\bcancel{c}]\times[\bcancel{c},c^{I}]\to[N/m,c^{I}]\ \text{——indexer 与主 KV 同源同长，省掉 CSA 那条从 } H \text{ 出发的压缩路径}
\end{aligned}$$

### Stage 4 · 本层 query 与 indexer query（三种模式都算 Q）

$$\begin{aligned}
c^{Q}_{t} &= h_{t}W^{DQ} && \text{// } [1,\bcancel{d}]\times[\bcancel{d},d_c]\to[1,1280]\ \text{——低秩 query latent，主注意力与 indexer 共用（结构沿用 CSA [INFERENCE]）}\\
q_{t} &= c^{Q}_{t}W^{UQ} && \text{// } [1,\bcancel{1280}]\times[\bcancel{1280},n_{h}c]\to[1,32768]\ \text{——} 64\times512\text{，每层自己的 global Q}\\
q^{I}_{t} &= c^{Q}_{t}W^{IUQ} && \text{// } [1,\bcancel{1280}]\times[\bcancel{1280},n^{I}_{h}c^{I}]\to[1,4096]\ \text{——} 32\times128\text{，仅 Full/Reindex 层需要}
\end{aligned}$$

### Stage 5 · 打分域随模式切换

$$\begin{aligned}
I_{t,s} &= \sum_{h=1}^{n^{I}_{h}}w^{I}_{t,h}\operatorname{ReLU}\!\left(q^{I,h}_{t}K^{I,Comp,l\top}_{s}\right) && \text{// } [n^{I}_{h},\bcancel{c^{I}}]\times[\bcancel{c^{I}},|\mathcal{D}|]\to[n^{I}_{h},|\mathcal{D}|]\xrightarrow{\text{按 } w^{I} \text{ 沿头维加权求和}} [|\mathcal{D}|]\ \text{——每前序条目一个标量分；} \mathcal{D} \text{ 在 Full 层为全部因果可见条目（编码器 } N/m\text{、解码器 } N\text{），在 Reindex 层为 } 16{,}384
\end{aligned}$$

### Stage 6 · 块级候选池（仅解码器首个 Full 层）

$$\begin{aligned}
b_{j} &= \max_{s\in\text{block}_{j}}I_{t,s} && \text{// } [N]\xrightarrow{\text{每 8 个位置取 max 池化}} [N/8]\ \text{——块分取块内最高分，抗单点噪声}\\
\mathcal{P}_{t} &= \bigcup_{j\in\operatorname{Top}_{2048}(b)}\text{block}_{j} && \text{// } [N/8]\xrightarrow{\text{沿块维 Top-2048}} [16{,}384]\ \text{——候选位置集合，建一次供后续 Reindex 层共用}
\end{aligned}$$

### Stage 7 · Top-$k$ 选择

$$\begin{aligned}
\mathcal{S}_{t} &= \operatorname{Top}_{k}\!\left(I_{t,\mathcal{D}}\right) && \text{// } [|\mathcal{D}|]\xrightarrow{\text{沿位置维 Top-}512} [512]\ \text{——Reuse 层跳过本阶段，直接沿用最近一次 Full/Reindex 的下标}
\end{aligned}$$

### Stage 8 · 选中条目加滑窗上做共享 KV 注意力

$$\begin{aligned}
\tilde{K} &= \left[\operatorname{Gather}\!\left(C^{Comp,l},\mathcal{S}_{t}\right);\,K^{win}_{l}\right] && \text{// } [N/m,c]\xrightarrow{\text{按 } \mathcal{S}_{t} \text{ 取行}}[512,c]\ \text{并上 } [128,c]\to[640,c]\ \text{——选中条目兼作 key 与 value，} n_{h} \text{ 头共用}\\
A_{t} &= \operatorname{Softmax}\!\left(q_{t}\tilde{K}^{\top}/\sqrt{c}+z'\right) && \text{// } [64,\bcancel{512}]\times[\bcancel{512},640]\to[64,640]\xrightarrow{\text{沿 } 640 \text{ 维归一}}[64,640]\ \text{——sink 项 } \operatorname{Exp}(z'_{h}) \text{ 计入分母}\\
o_{t} &= A_{t}\tilde{V} && \text{// } [64,\bcancel{640}]\times[\bcancel{640},512]\to[64,512]\ \text{——每个 head 一条 } c \text{ 维输出}
\end{aligned}$$

QK RMSNorm、partial RoPE、attention sink 与滑窗分支沿用 V4 的四个共用组件（§2.3.3）[^src-deepseek-v4-2606-19348]；V4.1 仅列出压缩器与 indexer 的简化，未声明移除这些组件 [INFERENCE]。论文未写缩放因子 $1/\sqrt{c}$，按惯例补 [INFERENCE]。

### Stage 9 · 分组输出投影

$$\begin{aligned}
o'^{G}_{t,i} &= o^{G}_{t,i}W^{G}_{i} && \text{// } [1,\bcancel{4096}]\times[\bcancel{4096},1024]\to[1,1024]\ \text{——每组 } (n_{h}/g)=8 \text{ 个头先压到 } d'_{g}\\
\hat{o}_{t} &= \operatorname{Concat}\!\left(o'^{G}_{t,1},\dots,o'^{G}_{t,8}\right)W^{O} && \text{// } [1,\bcancel{8192}]\times[\bcancel{8192},5120]\to[1,5120]\ \text{——拼接后回到残余流}
\end{aligned}$$

## 层配置与 KV 账

全模型 40 层：前 2 层纯 SWA；18 层编码器用 $m=2$，分 3 组各 6 层，每组首层 Full、其余 5 层 Reuse；20 层解码器用 $m=1$，分 5 组各 4 层，第 1 组首层 Full + 3 Reuse，后 4 组首层 Reindex + 3 Reuse（§4.2.1）[^src-deepseek-v4-1-2609-19969]。据此全模型仅 4 个 Full 层物化全局 KV（编码器 3 套 $m=2$、解码器 1 套 $m=1$）[^src-deepseek-v4-1-2609-19969]。

按 §2.4.4 的 FP4 格式反算每 token 字节数（[INFERENCE]，正文只给总数）：单条目 $512$ 通道 $\times4$ bit $=256$ B，加 $512/16=32$ 个 E4M3 scale 共 $288$ B；编码器 $3\times288/2=432$，解码器 $288$；indexer K 每条目 $128$ 通道 $=64$ B（含 scale 为 $72$ B），编码器 $3\times72/2=108$、解码器 $72$。合计 $900$ B（indexer K 不计 scale 则 $880$ B），与正文 890 bytes/token 相差约 1%（1.1%），说明「解码器只留一套全局 KV」自洽（§2.3.1、§2.4.4）[^src-deepseek-v4-1-2609-19969]。反推 V4-Flash 约 3.5 KB/token（$890\times4$）[INFERENCE]。

## 训练与推理侧适配

CSA2 跨流水线阶段共享，训练侧做三项处理：shadow indexer（每个参与阶段保留一份可执行副本，参数仍保持单逻辑 owner）、pipeline payload 扩展（跨阶段传递共享激活与稀疏路由信息，按上下文并行切分）、micro-batch 级共享状态管理（跟踪并发生效的 micro-batch，待最后一个消费者用完后释放）（§3.1.2）[^src-deepseek-v4-1-2609-19969]。推理侧将整层压入少数融合核：Reuse 层 prefill 15 个核、decode 11 个核（§3.2）[^src-deepseek-v4-1-2609-19969]。

## 证据

配置（§4.2.1）如「层配置与 KV 账」所列，另有 $n_{win}=128$、8 个输出分组、组内中间维 1024（Pro 侧无对照，本篇只发布 Flash）[^src-deepseek-v4-1-2609-19969]。效率：全局 KV 890 bytes/token，约为 V4-Flash 的 $1/4$、V1 的 $1/437$（摘要、Figure 1b、§6）[^src-deepseek-v4-1-2609-19969]；持久化 KV 为 V4-Flash 的 $1/8$，由「SWA KV 不落盘（约一半容量）」乘「全局 KV $1/4$」得出（§3.2.1）[^src-deepseek-v4-1-2609-19969]。4K 到 1M 的 decode 单 token FLOPs 仅增 $1/4$（Figure 2）[^src-deepseek-v4-1-2609-19969]。FP4 主 KV 的精度损失报告为「no measurable decrease」（§2.4.4）[^src-deepseek-v4-1-2609-19969]。

## 边界

CSA2 与 CSA/HCA 无同台消融，三种模式的层配比（各组首层 Full 或 Reindex、其余 Reuse）无对照实验，压缩器去除重叠与位置嵌入的代价未单独测量（§2.3、§4.2.1）[^src-deepseek-v4-1-2609-19969]。FP4 主 KV 的 QAT 仅在 post-training 阶段完成，精度结论仅给出「无显著下降」，未提供 BPB 或 benchmark 差值表（§2.4.4）[^src-deepseek-v4-1-2609-19969]。$k=512$、2,048 块、16,384 候选池与 $m=2/1$ 均无敏感性实验[^src-deepseek-v4-1-2609-19969]。HSI 的块大小 8 与 top-512 的关系（$512\times8=4{,}096<16{,}384$：Reindex 层选中位置仅够重建 4,096 处候选池）正文未讨论 [INFERENCE]。作者把「CSA2 的选择错误」列为未完全刻画的鲁棒性边界（§6）[^src-deepseek-v4-1-2609-19969]。与 V4 的效率对照属跨代对比，Figure 1b 仅给出 890 B 相对 V4-Flash 与 V1 的倍数且无细分项，降幅无法单独归因于 CSA2[^src-deepseek-v4-1-2609-19969]。

## 相关页面

[[compressed-sparse-attention]] 属于上一代（$1/4$ 重叠压缩 + DSA top-$k$），[[heavily-compressed-attention]] 重压缩支路在 V4.1 被 $m=1$ 的 CSA2 取代，[[deepseek-v4-hybrid-attention]] 记录被替换的混合架构；[[ced]] 是与 CSA2 配合的另一半改动（prefill 减半）；[[deepseek-sparse-attention]] 是 CSA 系复用的 indexer + top-$k$ 框架；[[native-sparse-attention]] 与 [[minimax-sparse-attention]] 是层内块选择的对照路线；[[kv-cache-compression]] 对应推理期缓存压缩谱系；[[source-deepseek-v4-1-2609-19969]] 与 [[source-deepseek-v4-2606-19348]] 对应两个源摘要。

[^src-deepseek-v4-1-2609-19969]: [[source-deepseek-v4-1-2609-19969]]
[^src-deepseek-v4-2606-19348]: [[source-deepseek-v4-2606-19348]]
