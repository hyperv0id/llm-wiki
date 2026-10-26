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
last_updated: 2026-10-27
source_count: 7
confidence: medium
status: active
---

# MiniMax Sparse Attention (MSA)

MiniMax（arXiv:2606.13392v2，2026-06）提出 GQA 原生块稀疏注意力：轻量 Index Branch 对 KV 块打分，为每个 GQA 组独立选 top-$k$ 块；Main Branch 只在选中块的因果可见 token 上做精确 softmax 注意力。相比标准 GQA 只增加索引 $Q/K$ 两个投影矩阵，支持从头稀疏预训练与 GQA checkpoint 转换两条路线（§1、§3、§3.2）[^src-minimax-sparse-attention-2606-13392]。

## 问题定位

超长上下文负载（agentic 工作流、仓库级代码推理、持久记忆）要求模型在数十万到百万 token 上联合注意，$\Theta(2H_qN^2d_h)$ 的注意力成本不可行（§1、§2.1）[^src-minimax-sparse-attention-2606-13392]。设计上按 Occam's razor 削减至必要组件，沿用 softmax 注意力范式以最大限度复用软硬件基础设施；块级选择搭配小 top-$k$，放宽 token 级细粒度设计对 head 维的约束，适配更广的 GPU（§1）[^src-minimax-sparse-attention-2606-13392]。同时，理论稀疏不等于实际加速：索引构建、top-$k$、反向索引、query gather 与负载均衡都有开销，需要算法与 GPU 执行路径协同设计（§1、§5.4）[^src-minimax-sparse-attention-2606-13392]。

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

式 9–10 的 KL 损失在选中 token 集上将索引分布对齐「教师」——即组内 $G$ 个主支路注意力分布在概率级的平均；$L_{KL}=\tfrac{1}{NH_{kv}}\sum_i\sum_r D_{KL}(\operatorname{stopgrad}(P_{i,\cdot}^{(r)})\,\|\,P^{idx,(r)}_{i,\cdot})$，教师 stopgrad。总损失 $\mathcal{L} = \mathcal{L}_{LM} + \lambda\sum_{\text{layers}} L_{KL}$，$\lambda$ 数值未给[^src-minimax-sparse-attention-2606-13392]。式 11 中索引输入走 $\operatorname{stopgrad}(X)$ 截断梯度，KL 梯度只更新 $W^{idx}_q, W^{idx}_k$，不进骨干[^src-minimax-sparse-attention-2606-13392]。索引预热（Indexer warmup）先以全注意力跑两支且只训索引投影，再切换稀疏；从头训练与 checkpoint 转换都用同一日程（§3.2），全规模实验 warmup 长 40B token（§5.1）[^src-minimax-sparse-attention-2606-13392]。包含 query 的 local block 固定占一个槽位；附录 C.2 消融显示，更激进的强制 sink 块与大固定局部窗在最终配方中都不需要，模型自己学出这两个模式（§C.2）[^src-minimax-sparse-attention-2606-13392]。

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

exp-free top-$k$ 利用 softmax 保序直接在原始分上排序选块，跳过 max/exp/sum；因 $B_k=128$、$k=16$ 组合落在通用 top-$k$ 内核弱区（radix 选择 amortize 在大 $B$、bitonic 为 $O(B\log^2 B)$），故每 warp 32 lane 各流 1/32 步长、共享内存维护 $k$ 元素 min-heap、经 $k$ 轮 shuffle 合并。Table 1（H800，fp32，(N,B) 输入，50 次中位延迟）：128K/1024/$k{=}16$ 时 779 µs 对 torch 3970（5.1×）与 TileLang 2864（3.7×）；512K/4096/$k{=}16$ 时 7880 对 33810（4.3×）与 17779（2.3×）；$k{=}32$ 档加速降至 1.2–2.7×[^src-minimax-sparse-attention-2606-13392]。KV-outer 遍历（式 13–16）中，Q-outer 的 FLOPs/IO 约 $G$，KV-outer 配 query gather 需中间输出缓冲且 FLOPs/IO 约 $\tfrac{2}{3}B_k$；因实践中 $\tfrac{2}{3}B_k\gg G$（85.3 对 16），故选 KV-outer：persistent grid 按 (kv_block, kv_head) 切瓦片，反向稀疏索引找出选中该块的 query 并由 TMA 拷入[^src-minimax-sparse-attention-2606-13392]。针对 sink 类热块（几乎每个 query 都选的早期块）使一 CTA 一瓦片失衡，调度 kernel 把每个 KV 瓦片沿 query 维切成至多约 $2kB_k$ 个 query 的块，热瓦片摊到多个 CTA 共享同一份 K/V 载入；每个 (query, chunk) 预分配槽位 $s\in[0,k)$ 并与 query 下标打包成 32-bit 句柄，注意力 kernel 直接写预定偏移，无需原子操作[^src-minimax-sparse-attention-2606-13392]。两阶段 forward 由 attention kernel 写局部归一化部分和 $O_{buf}\in\mathbb{R}^{k\times n\times H_q\times d}$ 与各部分 $\operatorname{LSE}$，combine kernel 按 split-K 权重 $w_s=\exp(\operatorname{LSE}_s-\operatorname{LSE}[i,h])$ 合并，两 kernel 借 Programmatic Dependent Launch 掩盖启动延迟[^src-minimax-sparse-attention-2606-13392]。query 拼接利用 KV-outer 下同一瓦片所有 gather 位置共享 K/V 操作数，内核把 $\lceil 128/G\rceil$ 个位置连同各组 $G$ 个 head 打包成 $128\times128$ 打分 MMA（Q-outer 因不同位置选的 KV 子集不同而无法实现）[^src-minimax-sparse-attention-2606-13392]。KL 损失 LSE 融合由主 pass 直接向全局内存输出 $LSE_{main}$ 与块级 $LSE_{idx}$，省去 KL 前向 kernel，反向直接读标量送入 softmax[^src-minimax-sparse-attention-2606-13392]。动态负载均衡则由 persistent grid 上 CTA 经全局原子计数器获取任务，子瓦片数随每瓦片 query 数伸缩（§4.3）[^src-minimax-sparse-attention-2606-13392]。

## 证据

### 实验设置（§5.1）

109B 总参/6B 激活 MoE，41 层（3 dense + 38 MoE），$d_{model}=3072$，注意力 64 query head / 4 KV head / 头维 128 / RoPE 64，MoE 128 routed + 1 shared、top-4；$B_k=128$、$k=16$。3T token 预算：MSA-PT 从头训练（40B warmup 后稀疏）；MSA-CPT 从 2.6T token 的 GQA checkpoint 换层后续训 400B（40B warmup + 稀疏）[^src-minimax-sparse-attention-2606-13392]。

### 训练动态（Figure 2、Figure 3）

3T 全程 MSA-PT 与 Full Attention 的 LM loss 曲线几乎重合（末 50B 窗口 1.21 对 1.22），梯度范数处于同区间（Figure 2）[^src-minimax-sparse-attention-2606-13392]。CPT 的 KL 在 warmup 期从约 0.40 快速下降，切至稀疏后保持低位；块 recall 与分数 recall 曲线处于 0.70–0.95 区间，分数 recall 高于块 recall（Figure 3，曲线逐点数值图内不可读）[^src-minimax-sparse-attention-2606-13392]。

### 下游评测（Table 2，3T 预算，Full / MSA-PT / MSA-CPT）

| 组 | 基准 | Full | MSA-PT | MSA-CPT |
|---|---|---|---|---|
| General | MMLU | 67.0 | 67.2 | 66.8 |
| General | MMLU-Pro | 38.5 | 38.8 | 39.1 |
| General | BBH | 67.7 | 66.6 | 66.1 |
| General | GPQA Hard | 25.9 | 26.3 | 26.3 |
| General | ARC Challenge | 82.7 | 82.5 | 82.9 |
| General | TriviaQA | 66.0 | 65.5 | 67.7 |
| General | WinoGrande | 58.3 | 60.9 | 62.0 |
| Math | GSM8K | 76.2 | 77.7 | 73.7 |
| Math | MGSM | 44.1 | 46.0 | 44.2 |
| Math | MathVista | 43.8 | 46.8 | 44.5 |
| Math | OlymMATH Easy P@100 | 23.0 | 26.0 | 22.0 |
| Code | HumanEval | 61.0 | 64.0 | 57.9 |
| Code | EvalPlus | 59.4 | 61.8 | 60.0 |
| Code | BigCodeBench | 44.8 | 44.0 | 45.7 |
| Code | MultiPL-E MBPP P@10 | 82.1 | 81.6 | 81.1 |
| Retrieval | RULER-8K | 79.8 | 84.2 | 77.2 |
| Retrieval | RULER-32K | 75.0 | 77.5 | 75.7 |
| Image | AI2D | 68.3 | 70.6 | 67.3 |
| Image | ChartQA | 75.0 | 75.4 | 71.4 |
| Image | MMMU | 46.8 | 45.9 | 44.5 |
| Image | OCRBench v2 | 55.0 | 55.7 | 54.3 |
| Image | CharXiv | 37.55 | 41.55 | 37.15 |
| Image | VisualWebBench | 55.6 | 68.4 | 59.4 |
| Image | CVBench | 57.0 | 59.7 | 58.8 |
| Video | EgoSchema | 29.6 | 37.6 | 25.8 |
| Video | LongVideoBench | 38.5 | 41.8 | 38.9 |
| Video | MLVU | 44.14 | 46.94 | 43.68 |
| Video | MMVU | 45.8 | 47.5 | 45.8 |
| Video | VideoMME | 41.11 | 45.48 | 39.65 |
| Video | TemporalBench | 49.4 | 53.4 | 50.6 |
| PPL ↓ | TAU2 | 1.155 | 1.148 | 1.150 |
| PPL ↓ | AgentCompany | 1.248 | 1.249 | 1.247 |
| PPL ↓ | HLE | 1.275 | 1.278 | 1.275 |
| PPL ↓ | SWE | 1.216 | 1.218 | 1.216 |

两个稀疏模型整体保持竞争力：MSA-PT 在多数数学、图像、视频与长上下文检索上最强，MSA-CPT 更保守，在文本/代码/PPL 上贴近 Full Attention（§5.3）[^src-minimax-sparse-attention-2606-13392]。

### 长上下文扩展（Table 3，MSA-CPT 再训约 140B 后）

| 基准 | 子集 | Full | MSA-CPT | Δ |
|---|---|---|---|---|
| HELMET-128K | Overall | 46.53 | 45.93 | −0.60 |
| HELMET-128K | ICL | 70.40 | 72.80 | +2.40 |
| HELMET-128K | Rerank/RAG | 34.60 | 32.50 | −2.10 |
| RULER-128K | Overall | 72.00 | 72.12 | +0.12 |
| RULER-128K | CWE/FWE | 46.35 | 45.00 | −1.35 |
| RULER-128K | MK/MQ/MV | 96.63 | 98.87 | +2.24 |
| RULER-128K | QA1/QA2 | 47.80 | 46.80 | −1.00 |
| RULER-128K | VT | 97.80 | 96.80 | −1.00 |

每 query 每组仍只看 $kB_k=2048$ 个 KV token，在紧预算下保持长上下文能力（§5.3）[^src-minimax-sparse-attention-2606-13392]。

### 效率（Figure 4，64/4 头、头维 128，H800）

在 1M 上下文处，每 token 注意力 FLOPs 降低 28.4×，实测 prefill 加速 14.2×（GQA 约 30 s 档）、decode 加速 7.6×（GQA 约 0.9 ms 档）。因索引构建、top-$k$、反向索引、query gather、负载均衡开销与更不规则的访存，运行时加速低于 FLOPs 降幅，差距随上下文变长而收敛[^src-minimax-sparse-attention-2606-13392]。

### 消融（附录 B/C）

梯度来源对比（B.2，10B 试点，Figure 7）显示，LM-only 保持短上下文但长程检索差，KL-only 提升检索却因去掉 $O^{idx}$ 削减注意力容量而使短上下文衰退，LM+KL 最为平衡；后续 C.3 显示 warmup 就位后可省去索引 value 头，最终配方仅保留 KL 监督（§C.3）[^src-minimax-sparse-attention-2606-13392]。KL 梯度隔离实验（B.3，Figure 8/9）中，不 detach 时大 KL 系数会在数百步内引发梯度尖峰而发散，稳定系数下短上下文基准渐进退化，源于骨干靠简化主支路分布降低 KL 的自蒸馏效应；detach 后两类失效均消除[^src-minimax-sparse-attention-2606-13392]。关于 warmup（B.4，Figure 10/11），主支路注意力熵在训练最初数百步骤降，零步开启稀疏使索引追赶快速移动的目标，warmup 则兼顾提升短上下文与长程检索[^src-minimax-sparse-attention-2606-13392]。可学习 sink 实验（B.5，Figure 12/13）表明，GPT-OSS 式 sink logit 未完全消除首 token sink，且 PPL 无一致收益，未进最终配方[^src-minimax-sparse-attention-2606-13392]。动态选择与滑窗对比（B.6，Figure 14）显示，FLOP 匹配的「首块 + 局部窗」固定模式在 agent 任务中 PPL 全程高于 MSA，同等预算下内容依赖选择优于位置固定模式（曲线数值图内不可读）[^src-minimax-sparse-attention-2606-13392]。在总选中 token 数不变的块长测试中（C.1，Table 4），PPL 基本持平，RULER-8K 为 72.5/72.8/73.8（$B_k$=32/64/128），RULER-32K 为 66.1/65.3/64.6，大块以有限质量损失换取了 kernel 效率[^src-minimax-sparse-attention-2606-13392]。针对强制 sink 与局部窗（C.2，Table 5），去掉强制后多数指标基本持平（MMLU 60.5 对 60.5、GSM8K 66.0 对 66.9、PPL 持平），但 RULER-32K 从 65.8 降至 61.5 差 4.3 分，正文仍记长程检索「comparable」，两处并列于此[^src-minimax-sparse-attention-2606-13392]。至于索引 value 头（C.3，Table 6），有无 value 头互有微弱优劣而无一方持续主导，最终为效率予以移除；推理期索引只需 $Q^{idx}{K^{idx}}^{\top}$ 的块内 max，完全绕开 value 聚合与指数（§C.3）[^src-minimax-sparse-attention-2606-13392]。

### 学到的模式（附录 A，Figure 5/6）

各 GQA 组都选局部对角与 sink 列，但长程条纹组间不同——索引学到的是组特异模式而非单一全局选择；且不强制时首 token 的 sink 现象在各层各 head 自然涌现[^src-minimax-sparse-attention-2606-13392]。

## 谱系

同库训练期稀疏注意力方法的分野（MSA 在 §6 将自身定位为同时覆盖 **per-GQA-group top-$k$ 共享 × 块级选择**两轴）[^src-minimax-sparse-attention-2606-13392]：

与 [[native-sparse-attention|NSA]] 相比，NSA 面向 MQA/MHA，压缩、选择与滑窗三支路各自 softmax 并经 sigmoid 门控合并，选择分复用压缩支路且组内求和；MSA 采用单主支路与强制 local block（无压缩或窗口支路），由专用索引头逐组独立选块，原生适配 GQA（§6）[^src-minimax-sparse-attention-2606-13392][^src-2502-11089-nsa]。与 [[moba-attention|MoBA]] 相比，MoBA 同样基于 GQA，但使用大 KV 块并按块内 key 均值打分，indexer 仅靠语言建模梯度训练；MSA 采用 $B_k=128$ 的块，由专用索引头配合 KL 对齐损失训练（§6）[^src-minimax-sparse-attention-2606-13392][^src-moba-mixture-of-block-attention]。与 [[deepseek-sparse-attention|DSA]] 相比，DSA 在 MLA 的 MQA 模式下逐 token 选 top-$k$，全部 query head 共享一份索引；MSA 逐块选取、每 GQA 组一份索引以保持 KV 连续读，两者 indexer 均保持 $O(N^2)$ 结构且在训练期对齐（DSA 采用 KL 两阶段 CPT）[^src-minimax-sparse-attention-2606-13392][^src-2512-02556-deepseek-v3-2]。与 [[compressed-sparse-attention|CSA]] 相比，CSA 先将 KV 压缩 4:1 再执行 DSA 式选择；MSA 不压缩 KV，直接在选中块的原始 token 上计算精确注意力[^src-minimax-sparse-attention-2606-13392][^src-deepseek-v4-2606-19348]。与 [[longformer-attention|Longformer]] 相比，Longformer 的滑窗、空洞与 global 均为内容无关的固定掩码；MSA 的支持集由学习所得索引按内容决定，B.6 的 FLOP 匹配滑窗对照即为该分野的实验版本[^src-minimax-sparse-attention-2606-13392][^src-beltagy-2020-longformer]。

与 NSA、MoBA、DSA 的训练路线差异：NSA 只演示从头训练[^src-2502-11089-nsa]；DSA 只演示 checkpoint 继续预训练[^src-2512-02556-deepseek-v3-2]；MSA 两条路都给（MSA-PT / MSA-CPT）并各自与同一 GQA 基线对照（§5）[^src-minimax-sparse-attention-2606-13392]。

## 边界

残差长程检索缺口需靠更长稀疏训练、更大推理时预算或更强打分函数解决（Outlook，§7）[^src-minimax-sparse-attention-2606-13392]。运行时加速低于 FLOPs 减少，索引、gather 与负载均衡等差距构成已列出但未逐项定量（§5.4）[^src-minimax-sparse-attention-2606-13392]。实验未与其他稀疏注意力方法同台，全部对照均采用自家的 GQA Full Attention 基线，与 NSA/MoBA/DSA 仅有 §6 的定性定位，唯一的稀疏结构对照是 10B 试点上的滑窗基线（B.6）[^src-minimax-sparse-attention-2606-13392]。附录消融多采用 10B 试点模型（B.1），C.1 明确使用更少迭代与评测子集，C.2 与 C.3 未说明模型规模[^src-minimax-sparse-attention-2606-13392]。$d_{idx}$ 与 $\lambda$ 数值未给，$k$ 无敏感性曲线（C.1 固定总预算只扫块长）[^src-minimax-sparse-attention-2606-13392]。MSA-CPT 在部分视频基准上低于 Full（EgoSchema 25.8 对 29.6），论文归因为基准依赖而非集中在单一能力域（§5.3）[^src-minimax-sparse-attention-2606-13392]。

## 相关页面

[[source-minimax-2026-sparse-attention]] 为源摘要；[[native-sparse-attention]] 为三支路及门控合并的块稀疏对照；[[moba-attention]] 为大块与块均值打分的 GQA 块稀疏对照；[[deepseek-sparse-attention]] 为 token 级 top-$k$、MQA 端点共享索引的对照；[[compressed-sparse-attention]] 为先压缩 KV 再跑 DSA 选择的对照；[[longformer-attention]] 为内容无关固定掩码端点；[[deepseek-v4-hybrid-attention]] 为压缩与选择分层交错路线；[[grouped-query-attention]] 为 MSA 依附的头共享结构；[[attention-sink]] 记录附录 A 与 B.5 观察及消融现象；[[flash-attention]] 为 MSA kernel 复用的算法骨架。线性注意力路线的对照端点见 [[kimi-delta-attention|KDA]]：Kimi Linear 报告将两条路线对立比较——稀疏注意力保留全 KV cache 做 token 选择、检索细但上限仍是 full attention，线性注意力以固定状态压缩换常数推理成本、细粒度选择弱但状态可学习（§7.1）[^src-moonshot-2025-kimi-linear]。

[^src-minimax-sparse-attention-2606-13392]: [[source-minimax-2026-sparse-attention]]
[^src-moonshot-2025-kimi-linear]: [[source-moonshot-2025-kimi-linear]]
[^src-2502-11089-nsa]: [[source-2502-11089-nsa]]
[^src-moba-mixture-of-block-attention]: [[source-moba-mixture-of-block-attention]]
[^src-2512-02556-deepseek-v3-2]: [[source-2512-02556-deepseek-v3-2]]
[^src-deepseek-v4-2606-19348]: [[source-deepseek-v4-2606-19348]]
[^src-beltagy-2020-longformer]: [[source-beltagy-2020-longformer]]
