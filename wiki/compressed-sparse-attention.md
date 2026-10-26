---
title: "Compressed Sparse Attention (CSA)"
type: technique
tags:
  - attention
  - sparse-attention
  - long-context
  - kv-cache
  - llm
created: 2026-10-27
last_updated: 2026-10-27
source_count: 5
confidence: medium
status: active
---

# Compressed Sparse Attention (CSA)

DeepSeek-V4 混合注意力的稀疏支路先将每 $m$ 个 token 的 KV cache 沿序列维压缩为 1 个 entry（净压缩率 $1/m$，两个模型均取 $m=4$），再在压缩条目上运行 [[deepseek-sparse-attention|DSA]]——lightning indexer 打分、逐 query 取 top-$k$、MQA 核心注意力（§2.3、§2.3.1）[^src-deepseek-v4-2606-19348]。V3.2 的 DSA 在原生长度上逐 token 选择 MLA latent，主注意力降至 $\mathcal{O}(Lk)$，但 indexer 打分仍为 $\mathcal{O}(L^2)$、KV cache 仍随全长增长（§2.3）[^src-2512-02556-deepseek-v3-2]；CSA 将选择对象与打分对象同时压短 $m$ 倍，两处开销同步降低（§2.3）[^src-deepseek-v4-2606-19348]。

## 压缩 KV：重叠双支路

给定隐藏状态 $H\in\mathbb{R}^{n\times d}$，CSA 先投影出两路 KV 条目与对应的压缩权重（式 9–10）[^src-deepseek-v4-2606-19348]：

$$C^{a}=H\cdot W^{KV}_{a},\quad C^{b}=H\cdot W^{KV}_{b},\quad Z^{a}=H\cdot W^{Z}_{a},\quad Z^{b}=H\cdot W^{Z}_{b}$$

其中 $C^{a},C^{b},Z^{a},Z^{b}\in\mathbb{R}^{n\times c}$，$c$ 为 head 维。第 $i$ 个压缩条目覆盖 token $[mi,\ m(i{+}1){-}1]$，除 $C^a$ 窗口外还叠加前一块尾部的 $C^b$ 窗口；两路拼接后做行 softmax，$2m$ 个权重一起归一化后加权求和（式 11–12）[^src-deepseek-v4-2606-19348]：

$$\begin{aligned}
\left[S^{a}_{mi:m(i+1)-1};\,S^{b}_{m(i-1):mi-1}\right] &= \operatorname{Softmax}_{\text{row}}\!\left(\left[Z^{a}_{mi:m(i+1)-1}+B^{a};\,Z^{b}_{m(i-1):mi-1}+B^{b}\right]\right)\\
C^{Comp}_{i} &= \sum_{j=mi}^{m(i+1)-1}S^{a}_{j}\odot C^{a}_{j}\;+\;\sum_{j=m(i-1)}^{mi-1}S^{b}_{j}\odot C^{b}_{j}
\end{aligned}$$

$\odot$ 是 Hadamard 积，$B^{a},B^{b}\in\mathbb{R}^{m\times c}$ 是可学习位置偏置。每个条目由 $2m$ 个原始条目算出，因 $C^b$ 在块 $i$ 与块 $i{-}1$ 的 $C^a$ 间重叠复用，每 $m$ 个新 token 仅净产出 1 个条目，序列压至 $1/m$（式 12 下方说明）[^src-deepseek-v4-2606-19348]。$i=0$ 时 $b$ 支路以 $-\infty$（权重）与 $0$（条目）填充。$c$ 各通道独立加权，压缩为逐通道凸组合 [INFERENCE]。

## Lightning indexer 与 top-k 选择

indexer keys 采用与 $C^{Comp}$ 相同的压缩压至 $\mathbb{R}^{(n/m)\times c^{I}}$（$c^{I}$ 为 indexer head 维），打分对象亦为压缩条目（§2.3.1）[^src-deepseek-v4-2606-19348]。query 侧经低秩 latent $c^{Q}_t=h_t\cdot W^{DQ}\in\mathbb{R}^{d_c}$ 升维产出 $n^{I}_h$ 个 indexer query（式 13–14），打分为逐头 ReLU 内积按头权重求和（式 15–16）：

$$I_{t,s}=\sum_{h=1}^{n^{I}_{h}}w^{I}_{t,h}\cdot\operatorname{ReLU}\!\left(q^{I,h}_{t}\cdot K^{I,Comp}_{s}\right),\qquad s<\lfloor t/m\rfloor$$

主注意力的 query latent 与 indexer query latent 共用 $c^{Q}_t$，主注意力 query 由 $c^{Q}_t\cdot W^{UQ}$ 升维得到（式 18）[^src-deepseek-v4-2606-19348]。打分后按式 17 取 top-$k$：$C^{SprsComp}_{t}=\{C^{Comp}_{s}\mid I_{t,s}\in\operatorname{Top-}k(I_{t,:})\}$，每个 query 仅保留 $k$ 个压缩条目。

## Shared-KV MQA 与分组输出投影

选中的压缩条目同时充当 key 与 value，$n_h$ 个 query head 共用同一组条目做 MQA（式 19）[^src-deepseek-v4-2606-19348]——这正是 [[multi-query-attention]] 的 $G=1$ 端点形态：一份 K/V 供 $H$ 个 query head 读取以摊薄访存（§2.4.1）[^src-shazeer-2019-mqa]。$c\cdot n_h$ 在 V4 中很大（$512\times 64$ 或 $512\times128$），直接投影回 $d$ 代价过高，于是将 $n_h$ 个头的输出分成 $g$ 组，每组先投到 $d'_g$ 维中间输出，拼接后再投至 $d$（§2.3.1 分组输出投影段）。SWA 分支的 $n_{win}$ 个未压缩条目也一并参与此次注意力计算（§2.3.3，见 [[deepseek-v4-hybrid-attention]]）。

## 数据流与维度变换

**维度符号**：$n$ 序列长度（token 数）；$d$ 隐藏维；$m=4$ 压缩率；$c=512$ head 维（亦为 KV entry 维与单头 query 维）；$k$ top-$k$（Flash 512 / Pro 1024，§4.2.1）；$n_h$ query head 数（64 / 128）；$d_c$ query 压缩维（1024 / 1536）；$n^{I}_h=64$ indexer 头数；$c^{I}=128$ indexer head 维；$g$ 输出分组数（8 / 16）；$d'_g=1024$ 组内中间维；$n_{win}=128$ 滑窗条目数（§4.2.1）[^src-deepseek-v4-2606-19348]。百万 token 上下文下有 $n/m\approx2.5\times10^{5}$ 个压缩条目，每 query 读取其中 $k$ 个与 $n_{win}$ 个滑窗条目。形状记 $[\text{行},\text{列}]$；论文未给出的实现细节标 [INFERENCE]。

### Stage 1 · 投出两路 KV 条目与压缩权重

$$\begin{aligned}
C^{a} &= H\,W^{KV}_{a} && \text{// } [n, \bcancel{d}]\times[\bcancel{d}, c]\to[n, c]\ \text{——每 token 一个 } a \text{ 路 KV entry，消去 } d\\
C^{b} &= H\,W^{KV}_{b} && \text{// } [n, \bcancel{d}]\times[\bcancel{d}, c]\to[n, c]\ \text{——} b \text{ 路同形，两套独立参数}\\
Z^{a},Z^{b} &= H\,W^{Z}_{a},H\,W^{Z}_{b} && \text{// } [n, \bcancel{d}]\times[\bcancel{d}, c]\to[n, c]\ \text{——逐 token 的压缩权重打分}
\end{aligned}$$

### Stage 2 · 重叠窗归一化与逐通道加权求和

$$\begin{aligned}
S^{ab}_{i} &= \operatorname{Softmax}_{\text{row}}\!\left(\left[Z^{a}_{mi:m(i+1)-1}+B^{a};\,Z^{b}_{m(i-1):mi-1}+B^{b}\right]\right) && \text{// } [2m, c] \xrightarrow{\text{沿 } 2m \text{ 维逐通道 softmax}} [2m, c]\ \text{——式(11)，} a \text{ 支路覆盖本块、} b \text{ 支路覆盖前块尾}\\
C^{Comp}_{i} &= \textstyle\sum_{j}S^{ab}_{i}[j]\odot\left[C^{a};C^{b}\right][j] && \text{// } [2m, c]\odot[2m, c] \xrightarrow{\text{沿 } 2m \text{ 维逐通道加权求和}} [c]\ \text{——式(12)，} c \text{ 每个通道各自做凸组合}\\
C^{Comp} &= \left\{C^{Comp}_{i}\right\}_{i=0}^{n/m-1} && \text{// } [n, c] \xrightarrow{\text{按 } m \text{ 分块压缩}} [n/m, c]\ \text{——} b \text{ 窗与邻块 } a \text{ 窗重叠，净压缩率仍 } 1/m
\end{aligned}$$

$i=0$ 的 $b$ 支路填充为 $-\infty$ 权重与 $0$ 条目，不贡献输出（式 11 下方）[^src-deepseek-v4-2606-19348]。

### Stage 3 · 压缩 indexer keys

$$\begin{aligned}
K^{I,Comp} &= \operatorname{Compress}\!\left(K^{I}\right) && \text{// } [n, c^{I}] \xrightarrow{\text{同一压缩算子}} [n/m, c^{I}]\ \text{——打分对象与主 KV 同步压短；未压缩 indexer key 的投影与该压缩的参数化正文未给 [INFERENCE]}
\end{aligned}$$

### Stage 4 · 低秩 query latent、indexer 打分与 top-k

$$\begin{aligned}
c^{Q}_{t} &= h_{t}\,W^{DQ} && \text{// } [1, \bcancel{d}]\times[\bcancel{d}, d_c]\to[1, d_c]\ \text{——式(13)，与主注意力共享的低秩 query latent}\\
q^{I}_{t} &= c^{Q}_{t}\,W^{IUQ} && \text{// } [1, \bcancel{d_c}]\times[\bcancel{d_c}, n^{I}_{h}c^{I}]\to[1, n^{I}_{h}c^{I}]\ \text{——式(14)，升维出 } n^{I}_{h} \text{ 个 indexer query}\\
I_{t,:} &= \textstyle\sum_{h}w^{I}_{t,h}\,\operatorname{ReLU}\!\left(q^{I,h}_{t}\,{K^{I,Comp}}^{\top}\right) && \text{// } [n^{I}_{h}, \bcancel{c^{I}}]\times[\bcancel{c^{I}}, n/m]\to[n^{I}_{h}, n/m] \xrightarrow{\text{按 } w^{I} \text{ 沿 } n^{I}_{h} \text{ 加权求和}} [n/m]\ \text{——式(15)(16)，每个前序压缩块一个标量分；} s \text{ 只取 } \lfloor t/m\rfloor \text{ 之前}\\
\mathcal{S}_{t} &= \operatorname{Top}_{k}\!\left(I_{t,:}\right) && \text{// } [n/m] \xrightarrow{\text{沿前序块维 Top-}k} [k]\ \text{——式(17)，} k \text{ 个压缩条目下标（取值 } 1..n/m\text{）}
\end{aligned}$$

### Stage 5 · 选中条目上做共享 KV MQA

$$\begin{aligned}
q_{t} &= c^{Q}_{t}\,W^{UQ} && \text{// } [1, \bcancel{d_c}]\times[\bcancel{d_c}, n_{h}c]\to[1, n_{h}c]\ \text{——式(18)，} n_{h} \text{ 个主注意力 head 与 indexer 同源}\\
\tilde{K} &= \operatorname{Concat}\!\left(\operatorname{Gather}\!\left(C^{Comp},\mathcal{S}_{t}\right),\,K^{win}\right) && \text{// } [n/m, c] \xrightarrow{\text{按 } \mathcal{S}_{t} \text{ 索引行}} [k, c]\ \text{再并上 } [n_{win}, c]\ \text{——滑窗条目未压缩，拼接进入同一次注意力 [INFERENCE]}\\
A_{t} &= \operatorname{softmax}\!\left(q_{t}\tilde{K}^{\top}/\sqrt{c}+z'\right) && \text{// } [n_{h}, \bcancel{c}]\times[\bcancel{c}, k+n_{win}]\to[n_{h}, k+n_{win}] \xrightarrow{\text{沿 } k+n_{win} \text{ 维归一化}} [n_{h}, k+n_{win}]\ \text{——QK 先过 RMSNorm，分母外加 sink 项 } \operatorname{Exp}(z'_{h})\\
o_{t} &= A_{t}\tilde{V} && \text{// } [n_{h}, \bcancel{k+n_{win}}]\times[\bcancel{k+n_{win}}, c]\to[n_{h}, c]\ \text{——压缩条目同时充当 key 与 value}
\end{aligned}$$

打分与加权中，CSA 与 HCA 共用 QK RMSNorm、attention sink 和滑窗分支，定义与动机见 [[deepseek-v4-hybrid-attention]]；缩放因子 $1/\sqrt{c}$ 论文未写明，按核心注意力惯例补 [INFERENCE]。

### Stage 6 · 分组输出投影

$$\begin{aligned}
o'^{G}_{t,i} &= o^{G}_{t,i}\,W^{G}_{i} && \text{// } [1, \bcancel{(n_{h}/g)c}]\times[\bcancel{(n_{h}/g)c}, d'_{g}]\to[1, d'_{g}]\ \text{——每组 } (n_{h}/g) \text{ 个头先压到 } d'_{g}\text{，Flash 为 } 4096\to1024\\
\hat{o}_{t} &= \operatorname{Concat}\!\left(o'^{G}_{t,1},\dots,o'^{G}_{t,g}\right)W^{O} && \text{// } [1, \bcancel{g\,d'_{g}}]\times[\bcancel{g\,d'_{g}}, d]\to[1, d]\ \text{——Flash } 8192\to4096\text{，Pro } 16384\to7168
\end{aligned}$$

分组将输出投影乘加数从 $c\,n_{h}\,d$ 降至 $(n_{h}/g)c\cdot d'_{g}+g\,d'_{g}\,d$；Pro 在 $d=7168$、$n_h=128$ 下直接投影约 $4.7\times10^{8}$ 乘加，分组后约 $1.2\times10^{8}$，论文仅给结构未列数值，按配置推算 [INFERENCE]。

## 相对 V3.2 DSA 的改动

同一实验室的两代稀疏注意力，改动集中在选择的对象与预算上：

选择粒度上，DSA 逐 token 选 MLA latent（$k=2048$）[^src-2512-02556-deepseek-v3-2]；CSA 逐压缩条目选（$m=4$ 压缩后 $k=512$ 或 $1024$），同等数量预算下覆盖的原始 token 范围扩大 $m$ 倍（§2.1）[^src-2512-02556-deepseek-v3-2][^src-deepseek-v4-2606-19348]。打分时，DSA 的 indexer 在原生长度上逐 token 打分，复杂度 $\mathcal{O}(L^2)$；CSA 的 indexer keys 先经同一压缩算子压到 $n/m$，打分矩阵随之缩短 $m$ 倍（§2.3.1）[^src-deepseek-v4-2606-19348]。DSA 的 indexer 与主注意力各自投影，CSA 则让主注意力 query 与 indexer query 共用低秩 latent $c^{Q}_t$，$W^{DQ}$ 只算一次（§2.3.1）[^src-deepseek-v4-2606-19348]。关于 top-$k$ 预算，论文把「相对 V3.2 选了更小的 top-$k$」列为短中等文本效率来源之一（§2.3.4）[^src-deepseek-v4-2606-19348]。

## 谱系定位

KV 共享上，CSA/HCA 落在 [[multi-query-attention]] 的 $G=1$ 端点：压缩条目仅一份，由 $n_h$ 个 query head 共用，[[grouped-query-attention]] 的组间共享在此退化为全共享（§2.3.1）[^src-shazeer-2019-mqa][^src-ainslie-2023-gqa]。MQA 压缩头维，CSA 进一步压缩序列维：每 query 读取 $n/m$ 中的 $k$ 个条目，而非全长。与 [[multi-head-latent-attention]] 对照，query 侧 $c^{Q}_t=h_t W^{DQ}$ 的低秩 latent 压缩与 MLA 的 query latent 同形 [INFERENCE]；KV 侧两者方向相反：MLA 沿特征维压 rank，CSA 沿序列维压 token 数。与 [[native-sparse-attention]] 的差异在于选择对象：NSA 在原始 token 的 64-token 块上选，CSA 在 $m=4$ 压缩出的条目上选，压缩步骤前置到 KV cache 本身（§3.3.2）[^src-2502-11089-nsa][^src-deepseek-v4-2606-19348]。

## 证据

配置（§4.2.1）上，Flash 采用 $m=4$、$n^{I}_h=64$、$c^{I}=128$、top-$k=512$，Pro 同前且为 top-$k=1024$，两者共享 $c=512$，Flash 为 $d_c=1024/n_h=64/g=8$、Pro 为 $d_c=1536/n_h=128/g=16$，$n_{win}=128$。FP4 QAT（§5.2.1）将 indexer QK 路径全置为 FP4（缓存、加载、乘法），index scores 由 FP32 转 BF16，top-$k$ selector 提速 2 倍且 KV entry recall 保持 99.7%。在 1M 上下文下（§2.3.4、Figure 1 右），Pro 的单 token FLOPs 为 V3.2 的 27%（低 3.7 倍）、KV cache 为 10%（小 9.5 倍），整体效率见 [[deepseek-v4-hybrid-attention]]，混合架构未单独拆分 CSA 的贡献。长上下文评测（Table 1）中，LongBench-V2 上 V4-Pro-Base 51.5 对 V3.2-Base 40.2，但该对照同时更换了数据、训练长度与全部架构改动，非 CSA 单独归因。

## 边界

论文缺少 CSA 与 V3.2 式逐 token DSA 的同台消融，未设「同模型只换选择粒度」对照，$1/m$ 压缩与小 $k$ 的组合收益未拆分（§2.3.1）[^src-deepseek-v4-2606-19348]。$k$、$m$、$n^{I}_h$、$c^{I}$ 均无敏感度实验，未说明取值依据（§4.2.1）[^src-deepseek-v4-2606-19348]。压缩权重 softmax 在 $2m$ 个元素上归一化的具体轴（逐通道或整块）正文仅有一句话，实现细节留给开源仓库（§2.3、脚注 1）[^src-deepseek-v4-2606-19348]。index scores 量化的 99.7% recall 属 QAT 后的数值，缺少量化前对照（§5.2.1）[^src-deepseek-v4-2606-19348]；检索类长程任务的失败模式（top-$k$ 漏选）未报告 recall 曲线，仅有端到端评测（Table 6、Figure 9）[^src-deepseek-v4-2606-19348]。

## 相关页面

[[deepseek-v4-hybrid-attention]] 包含 CSA 与 HCA 的交错混合、共用组件与效率总账；[[heavily-compressed-attention]] 是同一压缩算子的重度版，不选、全读；[[deepseek-sparse-attention]] 是 CSA 所复用的 indexer + top-k 框架（V3.2）。[[native-sparse-attention]] 作为块选择路线先例，选择对象是原始 token 块；[[multi-query-attention]] 是共享 KV 的 $G=1$ 形态来源，[[grouped-query-attention]] 处于头共享谱系中间档，CSA 位于 $G=1$ 端点。[[multi-head-latent-attention]] 为 query 低秩 latent 的同形对照（库内尚未建页，链接悬空），[[source-deepseek-v4-2606-19348]] 则是源摘要。

[^src-deepseek-v4-2606-19348]: [[source-deepseek-v4-2606-19348]]
[^src-2512-02556-deepseek-v3-2]: [[source-2512-02556-deepseek-v3-2]]
[^src-shazeer-2019-mqa]: [[source-shazeer-2019-mqa]]
[^src-ainslie-2023-gqa]: [[source-ainslie-2023-gqa]]
[^src-2502-11089-nsa]: [[source-2502-11089-nsa]]
