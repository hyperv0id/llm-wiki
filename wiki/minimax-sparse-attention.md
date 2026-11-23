---
title: "MiniMax Sparse Attention (MSA)"
type: technique
tags:
  - attention
  - sparse-attention
  - long-context
  - llm
  - gpu-kernel
created: 2026-10-27
last_updated: 2026-11-22
source_count: 7
confidence: medium
status: active
---

# MiniMax Sparse Attention (MSA)

长上下文要在数十万到百万 token 上联合注意，全量 softmax 的 $\mathcal{O}(N^2)$ 成本不可行；但稀疏化本身也不等于加速——打分、top-$k$、按索引 gather 与负载不均都会吃掉收益。MSA 的做法是给每个 GQA 组配一个轻量索引头，对 KV 块打分后独立选 top-$k$ 块，主支路只在选中块的因果可见 token 上做标准 softmax；相对标准 GQA 只多出索引 $Q/K$ 两个投影，算法与 GPU 执行路径配套设计，并同时支持从头稀疏预训练与 GQA checkpoint 转换[^src-minimax-sparse-attention-2606-13392]。

## 问题定位

设计上按 Occam's razor 砍到必要组件：沿用 softmax 注意力范式以复用软硬件基础设施，块级选择配小 top-$k$ 放宽 token 级细粒度对 head 维的约束，从而适配更广的 GPU。代价是索引构建、top-$k$、反向索引、query gather 与负载均衡各有开销，必须让算法与执行路径协同设计（§1、§2.1、§5.4）[^src-minimax-sparse-attention-2606-13392]。

## 机制

### Index Branch（式 5–7）

每个 GQA 组设一个索引 query 头，全部 GQA 组共享一个索引 key 头；对可见 token 打分后块内 max 池化，逐组独立取 top-$k$ 块：

$$\begin{aligned}
Q^{idx} &= XW^{idx}_q \in \mathbb{R}^{N\times H_{kv}\times d_{idx}},\qquad K^{idx} = XW^{idx}_k \in \mathbb{R}^{N\times 1\times d_{idx}}\\
S^{idx,(r)}_{i,j} &= Q^{idx,(r)}_i {K^{idx}_j}^{\top}/\sqrt{d_{idx}},\qquad M^{idx,(r)}_{i,b} = \max_{j\in\mathcal{B}_b,\ j\le i} S^{idx,(r)}_{i,j}\\
\mathcal{I}^{(r)}_i &= \operatorname{TopK}_b\!\left(M^{idx,(r)}_{i,\cdot},\ k\right)
\end{aligned}$$

$j\le i$ 施加因果；无可见 token 的块记为 $-\infty$；包含位置 $i$ 的 local block 在训练与推理中均入选（式 6–7）[^src-minimax-sparse-attention-2606-13392]。$\mathcal{I}^{(r)}_i$ 由组 $r$ 内 $G$ 个 query head 共用（§2.3）[^src-minimax-sparse-attention-2606-13392]。

### Main Branch（式 8）

组 $r$ 内任意 query head $h$ 的标准 scaled dot-product softmax 注意力限制在选中块的因果可见 token 上，每 query 预算至多 $kB_k$ 个 token，不随 $N$ 增长[^src-minimax-sparse-attention-2606-13392]：

$$O^{(h)}_i = \operatorname{softmax}\!\left(Q^{(h)}_i {K^{(r)}[\mathcal{I}^{(r)}_i]}^{\top}/\sqrt{d_h}\right) V^{(r)}[\mathcal{I}^{(r)}_i]$$

### 训练（§3.2、Algorithm 1）

top-$k$ 不可微，索引 $Q/K$ 投影无法获得语言建模损失梯度，训练信号与三项稳定设计为（§3.2）[^src-minimax-sparse-attention-2606-13392]：

KL 对齐损失在选中 token 集上把索引分布拉向「教师」——组内 $G$ 个主支路注意力分布在概率级的平均，教师 stopgrad：$L_{KL}=\tfrac{1}{NH_{kv}}\sum_i\sum_r D_{KL}(\operatorname{stopgrad}(P_{i,\cdot}^{(r)})\,\|\,P^{idx,(r)}_{i,\cdot})$。总损失 $\mathcal{L} = \mathcal{L}_{LM} + \lambda\sum_{\text{layers}} L_{KL}$，$\lambda$ 数值未给（式 9–10）[^src-minimax-sparse-attention-2606-13392]。索引输入走 $\operatorname{stopgrad}(X)$ 截断梯度，KL 梯度只更新 $W^{idx}_q, W^{idx}_k$，不进骨干（式 11）[^src-minimax-sparse-attention-2606-13392]。索引预热（Indexer warmup）先以全注意力跑两支且只训索引投影，再切换稀疏；从头训练与 checkpoint 转换都用同一日程（§3.2），全规模实验 warmup 长 40B token（§5.1）[^src-minimax-sparse-attention-2606-13392]。包含 query 的 local block 固定占一个槽位；附录 C.2 消融显示，更激进的强制 sink 块与大固定局部窗在最终配方中都不需要，模型自己学出这两个模式（§C.2）[^src-minimax-sparse-attention-2606-13392]。

### 复杂度（式 12）

$$F_{GQA}(N) = 2H_qd_hN^2,\qquad F_{MSA}(N) = \underbrace{H_{kv}d_{idx}N^2}_{\text{Index Branch}} + \underbrace{4H_qd_hNkB_k}_{\text{Main Branch}}$$

主支路预算 $kB_k$ 固定，索引支路系数 $H_{kv}d_{idx}\ll H_qd_h$（与 [[deepseek-sparse-attention|DSA]] 的 indexer 同样保持 $O(N^2)$ 结构但系数更小），$kB_k\ll N$ 时 FLOPs 差距随 $N$ 增长（§3.3）[^src-minimax-sparse-attention-2606-13392]。

## 数据流与维度变换

**维度符号**：$N$ 序列长度；$d_{model}$ 隐藏维（部署 3072）；$H_q$ query head 数（64）；$H_{kv}$ KV head 数（4）；$G=H_q/H_{kv}=16$ 每组 query head 数；$d_h$ 单头维（128）；$d_{idx}$ 索引单头维（无数值）；$B_k$ 块长（128）；$k$ 每组选中块数（16）；$B=\lceil N/B_k\rceil$ 块数（§5.1）[^src-minimax-sparse-attention-2606-13392]。形状 $[\text{行},\text{列}]$；训练期索引输入 detach（式 11）在 Stage 1 括注。

### Stage 1 · 索引打分（token 级 → 块级 max 池化）

输入 $X\in[N,d_{model}]$；训练期 $X$ 先过 stopgrad。

$$\begin{aligned}
Q^{idx} &= \operatorname{stopgrad}(X)W^{idx}_q && \text{// } [N, \bcancel{d_{model}}]\times[\bcancel{d_{model}}, H_{kv}d_{idx}]\to[N, H_{kv}, d_{idx}]\ \text{——式（5），每组一个索引 query 头}\\
K^{idx} &= \operatorname{stopgrad}(X)W^{idx}_k && \text{// } [N, \bcancel{d_{model}}]\times[\bcancel{d_{model}}, d_{idx}]\to[N, 1, d_{idx}]\ \text{——全部 GQA 组共享一个索引 key 头}\\
S^{idx,(r)} &= Q^{idx,(r)}{K^{idx}}^{\top}/\sqrt{d_{idx}} && \text{// } [N, \bcancel{d_{idx}}]\times[\bcancel{d_{idx}}, N]\to[H_{kv}, N, N]\ \text{——式（6），token 级打分，因果位置之外不计算}\\
M^{idx,(r)}_{i,b} &= \max_{j\in\mathcal{B}_b,\ j\le i} S^{idx,(r)}_{i,j} && \text{// } [H_{kv}, N, N] \xrightarrow{\text{沿块内可见 token 维 max 池化}} [H_{kv}, N, B]\ \text{——块分＝块内最高 token 分，无可见 token 记 } -\infty
\end{aligned}$$

### Stage 2 · 逐组 Top-$k$ 选块

$$\begin{aligned}
\mathcal{I}^{(r)}_i &= \operatorname{TopK}_b\!\left(M^{idx,(r)}_{i,\cdot},\ k\right)\ \cup\ \{\operatorname{local}(i)\} && \text{// } [H_{kv}, N, B] \xrightarrow{\text{沿 } B \text{ 维逐行取前 } k} [H_{kv}, N, k]\ \text{——式（7），local block 强制占一槽；组 } r \text{ 内 } G \text{ 个 query head 共用}
\end{aligned}$$

### Stage 3 · Main Branch 块稀疏精确注意力

选中块的因果可见 token 集 $\mathcal{I}^{tok,(r)}_i = \left(\bigcup_{b\in\mathcal{I}^{(r)}_i}\mathcal{B}_b\right)\cap\{1,\dots,i\}$，规模至多 $kB_k=2048$。

$$\begin{aligned}
K^{(r)}[\mathcal{I}^{(r)}_i],\, V^{(r)}[\mathcal{I}^{(r)}_i] &\leftarrow \operatorname{Gather} && \text{// } [d_h, N] \xrightarrow{\text{按 } \mathcal{I}^{tok,(r)}_i \text{ 索引列}} [d_h, \le kB_k]\ \text{——式（8），只取选中块内因果可见 token，预算与 } N \text{ 无关}\\
O^{(h)}_i &= \operatorname{softmax}\!\left(Q^{(h)}_i {K^{(r)}}^{\top}/\sqrt{d_h}\right)V^{(r)} && \text{// } [\bcancel{d_h}, \le kB_k]\times[\le kB_k, \bcancel{d_h}]\to[d_h]\ \text{——} h\in\mathsf{H}_r \text{ 各持自己的 query，标准 SDPA 限制在选中集合上}
\end{aligned}$$

### Stage 4 · KL 对齐（仅训练）

在 $\mathcal{I}^{tok,(r)}_i$ 上定义索引分布与教师（式 9）：

$$\begin{aligned}
P^{idx,(r)}_{i,\cdot} &= \operatorname{softmax}\!\left(S^{idx,(r)}_{i,\mathcal{I}^{tok}}\right) && \text{// } [\le kB_k] \xrightarrow{\text{沿选中 token 维归一化}} [\le kB_k]\ \text{——索引支路的选中集分布}\\
P^{(r)}_{i,\cdot} &= \tfrac{1}{G}\textstyle\sum_{\ell\in\mathsf{H}_r}\operatorname{softmax}\!\left(S^{(\ell)}_{i,\mathcal{I}^{tok}}\right) && \text{// } G\text{ 个主支路分布概率级平均}\to[\le kB_k]\ \text{——教师，与主支路梯度断开}\\
L_{KL} &= \tfrac{1}{NH_{kv}}\textstyle\sum_{i}\sum_{r}D_{KL}\!\left(\operatorname{stopgrad}(P^{(r)}_{i,\cdot})\,\|\,P^{idx,(r)}_{i,\cdot}\right) && \text{// } [N, H_{kv}] \xrightarrow{\text{逐位置逐组平均}} [1]\ \text{——式（10），梯度只达 } W^{idx}_q, W^{idx}_k
\end{aligned}$$

教师取概率级平均而非 logit 级，是论文的显式选择（§3.2「averages … at the probability level」）[^src-minimax-sparse-attention-2606-13392]。

## Kernel 设计（§4）

exp-free top-$k$ 利用 softmax 保序直接在原始分上排序选块，跳过 max/exp/sum；因 $B_k=128$、$k=16$ 组合落在通用 top-$k$ 内核弱区（radix 选择 amortize 在大 $B$、bitonic 为 $O(B\log^2 B)$），故每 warp 32 lane 各流 1/32 步长、共享内存维护 $k$ 元素 min-heap、经 $k$ 轮 shuffle 合并。H800 上的 fp32 中位延迟显示 128K 序列 779 µs 对 torch 的 3970（5.1×）、TileLang 的 2864（3.7×），512K 时 7880 对 33810（4.3×），$k{=}32$ 档加速降到 1.2–2.7×（Table 1）[^src-minimax-sparse-attention-2606-13392]。两种遍历中，Q-outer 的 FLOPs/IO 约 $G$，KV-outer 配 query gather 需中间输出缓冲且 FLOPs/IO 约 $\tfrac{2}{3}B_k$（式 13–16）；因实践中 $\tfrac{2}{3}B_k\gg G$（85.3 对 16），故选 KV-outer：persistent grid 按 (kv_block, kv_head) 切瓦片，反向稀疏索引找出选中该块的 query 并由 TMA 拷入[^src-minimax-sparse-attention-2606-13392]。针对 sink 类热块（几乎每个 query 都选的早期块）使一 CTA 一瓦片失衡，调度 kernel 把每个 KV 瓦片沿 query 维切成至多约 $2kB_k$ 个 query 的块，热瓦片摊到多个 CTA 共享同一份 K/V 载入；每个 (query, chunk) 预分配槽位 $s\in[0,k)$ 并与 query 下标打包成 32-bit 句柄，注意力 kernel 直接写预定偏移，无需原子操作[^src-minimax-sparse-attention-2606-13392]。两阶段 forward 由 attention kernel 写局部归一化部分和 $O_{buf}\in\mathbb{R}^{k\times n\times H_q\times d}$ 与各部分 $\operatorname{LSE}$，combine kernel 按 split-K 权重 $w_s=\exp(\operatorname{LSE}_s-\operatorname{LSE}[i,h])$ 合并，两 kernel 借 Programmatic Dependent Launch 掩盖启动延迟[^src-minimax-sparse-attention-2606-13392]。query 拼接利用 KV-outer 下同一瓦片所有 gather 位置共享 K/V 操作数，内核把 $\lceil 128/G\rceil$ 个位置连同各组 $G$ 个 head 打包成 $128\times128$ 打分 MMA（Q-outer 因不同位置选的 KV 子集不同而无法实现）[^src-minimax-sparse-attention-2606-13392]。KL 损失 LSE 融合由主 pass 直接向全局内存输出 $LSE_{main}$ 与块级 $LSE_{idx}$，省去 KL 前向 kernel，反向直接读标量送入 softmax[^src-minimax-sparse-attention-2606-13392]。动态负载均衡则由 persistent grid 上 CTA 经全局原子计数器获取任务，子瓦片数随每瓦片 query 数伸缩（§4.3）[^src-minimax-sparse-attention-2606-13392]。

## 证据

1. **定位：以更低开销逼近自家 Full Attention，不是在精度上超越它。** 3T token 预算下 MSA-PT 与 GQA Full Attention 的训练 LM loss 曲线几乎重合（末 50B 窗口 1.21 对 1.22），下游多数数学、图像、视频与长上下文检索指标略高（视频侧提升最明显，EgoSchema 29.6→37.6、VideoMME 41.11→45.48），MSA-CPT 更保守、在文本与代码及 PPL 上贴近 Full；1M 上下文每 token 注意力 FLOPs 降 28.4×，实测 prefill 加速 14.2×、decode 7.6×，索引构建、top-$k$、gather 与负载均衡的开销让实测加速低于 FLOPs 降幅（Figure 2、4）[^src-minimax-sparse-attention-2606-13392]。
2. **归因：KL 对齐与索引预热是必需件，强制局部块主要保长程检索。** 梯度来源对比里 LM-only 保持短上下文但长程检索差，KL-only 提升检索却让短上下文衰退，两者合用最平衡；KL 不 detach 时骨干靠简化主支路分布自蒸馏，数百步内梯度尖峰发散，稳定系数下则短上下文渐进退化；warmup 就位后可去掉索引 value 头。去掉强制的 sink 块与局部窗后 MMLU、GSM8K 基本持平，但 RULER-32K 从 65.8 掉到 61.5；FLOP 匹配的「首块 + 局部窗」固定模式在 agent 任务上 PPL 全程高于 MSA。不强制时首 token 的 sink 与局部对角仍在各层自然涌现，索引学到的是组特异模式（附录 A、B.2–B.6、C.2–C.3）[^src-minimax-sparse-attention-2606-13392]。
3. **缺口与负结果：没有与其他稀疏注意力方法同台。** 全部对照都是自家 GQA Full Attention，与 NSA、MoBA、DSA 只有定性定位，唯一的稀疏结构对照是 10B 试点上的滑窗基线；消融多在 10B 试点，部分用更少迭代与评测子集，$d_{idx}$ 与 $\lambda$ 数值未给，$k$ 无敏感性曲线。长上下文扩展的收益也薄：MSA-CPT 在 HELMET-128K 整体 −0.6、RULER-128K +0.12，子项互有胜负；视频侧 EgoSchema 25.8 对 Full 的 29.6 明显落后（附录 B.1、C.1、Table 3）[^src-minimax-sparse-attention-2606-13392]。

## 谱系

同库训练期稀疏注意力方法的分野（MSA 在 §6 将自身定位为同时覆盖 **per-GQA-group top-$k$ 共享 × 块级选择**两轴）[^src-minimax-sparse-attention-2606-13392]：

与 [[native-sparse-attention|NSA]] 相比，NSA 面向 MQA/MHA，压缩、选择与滑窗三支路各自 softmax 并经 sigmoid 门控合并，选择分复用压缩支路且组内求和；MSA 采用单主支路与强制 local block（无压缩或窗口支路），由专用索引头逐组独立选块，原生适配 GQA（§6）[^src-minimax-sparse-attention-2606-13392][^src-2502-11089-nsa]。与 [[moba-attention|MoBA]] 相比，MoBA 同样基于 GQA，但使用大 KV 块并按块内 key 均值打分，indexer 仅靠语言建模梯度训练；MSA 采用 $B_k=128$ 的块，由专用索引头配合 KL 对齐损失训练（§6）[^src-minimax-sparse-attention-2606-13392][^src-moba-mixture-of-block-attention]。与 [[deepseek-sparse-attention|DSA]] 相比，DSA 在 MLA 的 MQA 模式下逐 token 选 top-$k$，全部 query head 共享一份索引；MSA 逐块选取、每 GQA 组一份索引以保持 KV 连续读，两者 indexer 均保持 $O(N^2)$ 结构且在训练期对齐（DSA 采用 KL 两阶段 CPT）[^src-minimax-sparse-attention-2606-13392][^src-2512-02556-deepseek-v3-2]。与 [[compressed-sparse-attention|CSA]] 相比，CSA 先将 KV 压缩 4:1 再执行 DSA 式选择；MSA 不压缩 KV，直接在选中块的原始 token 上计算精确注意力[^src-minimax-sparse-attention-2606-13392][^src-deepseek-v4-2606-19348]。与 [[longformer-attention|Longformer]] 相比，Longformer 的滑窗、空洞与 global 均为内容无关的固定掩码；MSA 的支持集由学习所得索引按内容决定，B.6 的 FLOP 匹配滑窗对照即为该分野的实验版本[^src-minimax-sparse-attention-2606-13392][^src-beltagy-2020-longformer]。

与 NSA、MoBA、DSA 的训练路线差异：NSA 只演示从头训练[^src-2502-11089-nsa]；DSA 只演示 checkpoint 继续预训练[^src-2512-02556-deepseek-v3-2]；MSA 两条路都给（MSA-PT / MSA-CPT）并各自与同一 GQA 基线对照（§5）[^src-minimax-sparse-attention-2606-13392]。

## 边界

残差长程检索缺口需靠更长稀疏训练、更大推理时预算或更强打分函数解决（Outlook，§7）[^src-minimax-sparse-attention-2606-13392]。运行时加速低于 FLOPs 减少，索引、gather 与负载均衡等差距构成已列出但未逐项定量（§5.4）[^src-minimax-sparse-attention-2606-13392]。实验未与其他稀疏注意力方法同台，全部对照均采用自家的 GQA Full Attention 基线，与 NSA/MoBA/DSA 仅有 §6 的定性定位，唯一的稀疏结构对照是 10B 试点上的滑窗基线（B.6）[^src-minimax-sparse-attention-2606-13392]。附录消融多采用 10B 试点模型（B.1），C.1 明确使用更少迭代与评测子集，C.2 与 C.3 未说明模型规模[^src-minimax-sparse-attention-2606-13392]。$d_{idx}$ 与 $\lambda$ 数值未给，$k$ 无敏感性曲线（C.1 固定总预算只扫块长）[^src-minimax-sparse-attention-2606-13392]。论文把 MSA-CPT 的视频侧落后归因为基准依赖而非集中在单一能力域（§5.3）[^src-minimax-sparse-attention-2606-13392]。

## 相关页面

[[source-minimax-2026-sparse-attention]] 为源摘要；[[native-sparse-attention]] 为三支路及门控合并的块稀疏对照；[[moba-attention]] 为大块与块均值打分的 GQA 块稀疏对照；[[deepseek-sparse-attention]] 为 token 级 top-$k$、MQA 端点共享索引的对照；[[compressed-sparse-attention]] 为先压缩 KV 再跑 DSA 选择的对照；[[longformer-attention]] 为内容无关固定掩码端点；[[deepseek-v4-hybrid-attention]] 为压缩与选择分层交错路线；[[grouped-query-attention]] 为 MSA 依附的头共享结构；[[attention-sink]] 记录附录 A 与 B.5 观察及消融现象；[[flash-attention]] 为 MSA kernel 复用的算法骨架。线性注意力路线的对照端点见 [[kimi-delta-attention|KDA]]：Kimi Linear 报告将两条路线对立比较——稀疏注意力保留全 KV cache 做 token 选择、检索细但上限仍是 full attention，线性注意力以固定状态压缩换常数推理成本、细粒度选择弱但状态可学习（§7.1）[^src-moonshot-2025-kimi-linear]。

[^src-minimax-sparse-attention-2606-13392]: [[source-minimax-2026-sparse-attention]]
[^src-moonshot-2025-kimi-linear]: [[source-moonshot-2025-kimi-linear]]
[^src-2502-11089-nsa]: [[source-2502-11089-nsa]]
[^src-moba-mixture-of-block-attention]: [[source-moba-mixture-of-block-attention]]
[^src-2512-02556-deepseek-v3-2]: [[source-2512-02556-deepseek-v3-2]]
[^src-deepseek-v4-2606-19348]: [[source-deepseek-v4-2606-19348]]
[^src-beltagy-2020-longformer]: [[source-beltagy-2020-longformer]]
