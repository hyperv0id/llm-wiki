---
title: "Heavily Compressed Attention (HCA)"
type: technique
tags:
  - attention
  - long-context
  - kv-cache
  - llm
  - compression
created: 2026-10-27
last_updated: 2026-10-27
source_count: 3
confidence: medium
status: active
---

# Heavily Compressed Attention (HCA)

DeepSeek-V4 混合注意力的重压缩支路将每 $m'$ 个 token 的 KV cache 合并为 1 个 entry（$m'=128$，约为 CSA 压缩率 $m=4$ 的 32 倍），且**不做任何稀疏选择**，对全部压缩条目计算 dense 注意力（§2.3、§2.3.2）[^src-deepseek-v4-2606-19348]。其与 CSA 的分工为：CSA 用轻度压缩配合 top-$k$ 选择保留细粒度，HCA 用重度压缩换取全覆盖——每个前序 token 均参与注意力，分辨率压至 $1/128$；百万 token 上下文在 HCA 层只剩约 $8\times10^{3}$ 个条目（按 $2^{20}$ token 计为 8,192 [INFERENCE]）。

## 压缩 KV：非重叠单路

压缩策略与 CSA 大体相同，但压缩率更高且不重叠（§2.3.2）[^src-deepseek-v4-2606-19348]。单路投影生成条目与权重（式 20–21），经块内 softmax 后逐通道加权求和（式 22–23）：

$$\begin{aligned}
C &= H\cdot W^{KV},\qquad Z = H\cdot W^{Z}\\
S_{m'i:m'(i+1)-1} &= \operatorname{Softmax}_{\text{row}}\!\left(Z_{m'i:m'(i+1)-1}+B\right)\\
C^{Comp}_{i} &= \sum_{j=m'i}^{m'(i+1)-1}S_{j}\odot C_{j}
\end{aligned}$$

$C,Z\in\mathbb{R}^{n\times c}$，$B\in\mathbb{R}^{m'\times c}$ 是可学习位置偏置。与 CSA 的差异主要有两点：一是仅设单支路且窗口不重叠，每 $m'$ 个 token 产出 1 个条目，序列压缩为 $1/m'$；二是无 $b$ 支路的跨块重叠，块边界处的 token 不会同时出现在两个条目中（§2.3.2）[^src-deepseek-v4-2606-19348]。

query 侧、MQA 及分组输出投影与 CSA 完全同构：低秩 latent $c^{Q}_t=h_t W^{DQ}$ 升维出 $n_h$ 个 query；压缩条目兼作 key 与 value，由 $n_h$ 个 head 共用；输出按 $g$ 组先投到 $d'_g$ 再投到 $d$（§2.3.2 Shared Key-Value MQA and Grouped Output Projection）[^src-deepseek-v4-2606-19348]。逐公式维度见 [[compressed-sparse-attention]] Stage 5–6；SWA 分支、QK RMSNorm、partial RoPE 与 attention sink 同样适用（§2.3.3，见 [[deepseek-v4-hybrid-attention]]）。

## 数据流与维度变换

**维度符号**：$n$ 序列长度；$d$ 隐藏维；$m'=128$ 压缩率；$c=512$ head 维（KV entry 维与单头 query 维）；$n_h$ query head 数（Flash 64 / Pro 128）；$d_c$ query 压缩维（1024 / 1536）；$g$ 输出分组数（8 / 16）；$d'_g=1024$；$n_{win}=128$ 滑窗条目数（§4.2.1）[^src-deepseek-v4-2606-19348]。百万 token 下每 query 读 $n/m' + n_{win}$ 个条目，按 $2^{20}$ token 计为 $8{,}192+128=8{,}320$ [INFERENCE]；其余未注明的实现细节同标 [INFERENCE]。

### Stage 1 · 非重叠压缩

$$\begin{aligned}
C &= H\,W^{KV} && \text{// } [n, \bcancel{d}]\times[\bcancel{d}, c]\to[n, c]\ \text{——每 token 一个 KV entry，单路，消去 } d\\
Z &= H\,W^{Z} && \text{// } [n, \bcancel{d}]\times[\bcancel{d}, c]\to[n, c]\ \text{——逐 token 压缩权重打分}\\
S_{i} &= \operatorname{Softmax}_{\text{row}}\!\left(Z_{m'i:m'(i+1)-1}+B\right) && \text{// } [m', c] \xrightarrow{\text{沿 } m' \text{ 维逐通道 softmax}} [m', c]\ \text{——式(22)，块内归一化、不跨块}\\
C^{Comp} &= \left\{\textstyle\sum_{j}S_{i}[j]\odot C[j]\right\}_{i=0}^{n/m'-1} && \text{// } [m', c]\odot[m', c] \xrightarrow{\text{沿 } m' \text{ 维逐通道加权求和}} [n/m', c]\ \text{——式(23)，每 } m' \text{ 个 token 产出 } 1 \text{ 个条目，百万 token 约压到 } 8\times10^{3}
\end{aligned}$$

### Stage 2 · 低秩 query

$$\begin{aligned}
c^{Q}_{t} &= h_{t}\,W^{DQ} && \text{// } [1, \bcancel{d}]\times[\bcancel{d}, d_c]\to[1, d_c]\ \text{——式(24)，query 先压到低秩 latent}\\
q_{t} &= c^{Q}_{t}\,W^{UQ} && \text{// } [1, \bcancel{d_c}]\times[\bcancel{d_c}, n_{h}c]\to[1, n_{h}c]\ \text{——式(25)，} n_{h} \text{ 个 head 从同一 latent 升维}
\end{aligned}$$

### Stage 3 · 对全部压缩条目做 dense MQA

$$\begin{aligned}
\tilde{K} &= \operatorname{Concat}\!\left(C^{Comp},\,K^{win}\right) && \text{// } [n/m', c]\oplus[n_{win}, c]\to[n/m'+n_{win}, c]\ \text{——无选择器，全量进入注意力；拼接顺序 [INFERENCE]}\\
A_{t} &= \operatorname{softmax}\!\left(q_{t}\tilde{K}^{\top}/\sqrt{c}+z'\right) && \text{// } [n_{h}, \bcancel{c}]\times[\bcancel{c}, n/m'+n_{win}]\to[n_{h}, n/m'+n_{win}] \xrightarrow{\text{沿 } n/m'+n_{win} \text{ 维归一化}} [n_{h}, n/m'+n_{win}]\ \text{——QK 先过 RMSNorm，sink 项 } \operatorname{Exp}(z'_{h}) \text{ 入分母}\\
o_{t} &= A_{t}\tilde{V} && \text{// } [n_{h}, \bcancel{n/m'+n_{win}}]\times[\bcancel{n/m'+n_{win}}, c]\to[n_{h}, c]\ \text{——压缩条目同时充当 key 与 value}
\end{aligned}$$

$1/\sqrt{c}$ 缩放按核心注意力惯例补齐 [INFERENCE]。

### Stage 4 · 分组输出投影

与 CSA 的 Stage 6 相同（式 19 段后的分组输出投影）：$n_h$ 个头分 $g$ 组，每组 $(n_h/g)c\to d'_g$、拼接后 $g\,d'_g\to d$，维度对应见 [[compressed-sparse-attention]]。

## 在混合架构中的位置

CSA 与 HCA 交错组成 V4 的注意力层，且分工不对称：Pro 前两层全用 HCA（无选择器的全覆盖层），Flash 前两层用纯滑窗注意力，其后两层交错（§4.2.1）[^src-deepseek-v4-2606-19348]。将「重压缩」与「稀疏选择」拆到不同层，与 [[native-sparse-attention]] 在同层门控合并压缩、选择与滑窗是两种路线：NSA 每层三分支各自 softmax 后按门控相加（§3.2）[^src-2502-11089-nsa]；V4 每层仅有一种注意力形态，由交错排布承担层间互补。代价是信息难以兼顾：HCA 层覆盖全但条目粗糙，CSA 层清晰但仅覆盖 $k/(n/m)$ 的条目；论文未给出两类层的效果对比或占比消融（§4.2.1）[^src-deepseek-v4-2606-19348]。

KV cache 中，HCA 层每 token 分摊 $c/m'$ 个 entry 维，是 CSA 层（$c/m$）的 $1/32$；V4 的 KV cache 在 1M 上下文下约为 BF16 GQA8（head dim 128）基线的 2%（§2.3.4）[^src-deepseek-v4-2606-19348]。缓存布局按 $\operatorname{lcm}(m,m')=128$ token 对齐，每块产出 32 个 CSA 条目与 1 个 HCA 条目（§3.5.1、Figure 6，见 [[deepseek-v4-hybrid-attention]]）。

后继模型不再保留这条支路：DeepSeek-V4.1-Flash 用纯 [[csa2|CSA2]] 取代 CSA–HCA 混合，所有全局注意力层都是 top-$k$ 稀疏，没有对全部条目做 dense 的层；原本由 HCA 承担的全覆盖只保留在 indexer 的全长打分里，解码器后续层则被限制在 16,384 位置的候选池内（§1、§2.3、§2.3.2）[^src-deepseek-v4-1-2609-19969]。

## 证据

配置（§4.2.1）涵盖两个模型 $m'=128$：Flash 43 层、Pro 61 层，CSA/HCA 交错；Pro 前两层为 HCA、Flash 前两层为纯 SWA；head 维 $c=512$、$n_{win}=128$ 与 CSA 一致。效率（§2.3.4、Figure 1 右）上，1M 上下文下 Flash 单 token FLOPs 为 V3.2 的 10%（低 9.8 倍）、KV cache 为 7%（小 13.7 倍），Pro 为 27% / 10%，HCA 与 CSA 的相对贡献未拆分。长上下文质量（Table 6、Table 7）上，MRCR 1M 上 Pro-Max 83.5、Flash-Max 78.7，CorpusQA 1M 上 62.0 / 60.5，均为整模成绩，HCA 层的独立贡献无消融。

## 边界

论文未做 $m'=128$ 与「HCA 层在总层数中位置/占比」的敏感度测试，未给出交错的具体排布（逐层交替还是分段）（§4.2.1）[^src-deepseek-v4-2606-19348]。缺少 HCA 对 CSA 的替代实验，即不存在「全 CSA」「全 HCA」「只调比例」的对照，未在内部验证混合本身是否必要（§2.3）[^src-deepseek-v4-2606-19348]。$1/128$ 压缩的信息损失仅体现在端到端评测，未测量压缩条目的重建质量与 per-entry 信息保留度（§2.3.2）[^src-deepseek-v4-2606-19348]。Pro 前两层用 HCA、Flash 前两层用纯 SWA 的不对称设计无论证（§4.2.1）[^src-deepseek-v4-2606-19348]。

## 相关页面

[[compressed-sparse-attention]] 是同一压缩算子的轻度版加 top-k 稀疏选择；[[deepseek-v4-hybrid-attention]] 涉及交错混合、共用组件与效率总账；[[deepseek-sparse-attention]] 为 CSA 侧引用的 DSA 框架，HCA 不使用它；[[native-sparse-attention]] 为单层三分支门控合并的对照路线；[[multi-query-attention]] 是共享 KV 的 $G=1$ 形态来源；[[grouped-query-attention]] 为头共享谱系背景；[[multi-head-latent-attention]] 是 query 低秩 latent 的同形对照（库内尚未建页，链接悬空）；[[source-deepseek-v4-2606-19348]] 为源摘要。

[^src-deepseek-v4-2606-19348]: [[source-deepseek-v4-2606-19348]]
[^src-2502-11089-nsa]: [[source-2502-11089-nsa]]
[^src-deepseek-v4-1-2609-19969]: [[source-deepseek-v4-1-2609-19969]]
