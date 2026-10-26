---
title: "Lightning Attention：线性注意力的 IO-aware 分块实现"
type: technique
tags:
  - attention
  - linear-attention
  - io-aware
  - long-context
  - llm
created: 2026-10-27
last_updated: 2026-10-27
source_count: 2
confidence: medium
status: active
---

# Lightning Attention

Lightning attention 是 TransNormer（线性注意力变体，Qin et al. 2022）的 I/O-aware 优化实现：将因果注意力拆为块内左积与块间右积两段计算，绕开因果语言建模中缓慢且不可并行的 cumsum 操作，复杂度 $O(nd^2+nBd)$（§2.2.1）[^src-minimax-2025-01]。MiniMax-01 以「每 7 个 lightning attention 块配 1 个 softmax attention 块」的 hybrid 配置将其推至 456B 基础模型规模，实现「the first successful large-scale implementation of linear attention」（§1）[^src-minimax-2025-01]。

## 问题：causal 线性注意力的 cumsum 瓶颈

线性注意力用「右积核技巧」重排注意力：将 $\mathbf{O}=\mathrm{Norm}((\mathbf{Q}\mathbf{K}^{\top})\mathbf{V})$（式 2）改写为 $\mathbf{O}=\mathrm{Norm}(\mathbf{Q}(\mathbf{K}^{\top}\mathbf{V}))$（式 3），把二次复杂度降为线性；训练 $O(nd^2)$，推理时靠递推更新 $\mathbf{K}^{\top}\mathbf{V}$ 达到常数 $O(d^2)$/步（§2.2）[^src-minimax-2025-01]。但因果掩码下右积失效，须改算 cumsum，"This limitation impedes the realization of highly efficient parallel computation"，这也解释了提出九年后 LLaMA3、Qwen2.5、DeepSeek-V3、Mistral 为何均未采用线性注意力（§2.2）[^src-minimax-2025-01]。

## 机制：块内左积 + 块间右积

Lightning attention 的推导省略了 normalization、SiLU 激活与 gating（§2.2.1）[^src-minimax-2025-01]。左积形式为 $\mathbf{O}=[(\mathbf{Q}\mathbf{K}^{\top})\odot\mathbf{M}]\mathbf{V}$（式 4，$\mathbf{M}_{ts}=1$ iff $t\ge s$）；右积递推 $\mathbf{kv}_t=\mathbf{kv}_{t-1}+\mathbf{k}_t\mathbf{v}_t^{\top}$、$\mathbf{o}_t^{\top}=\mathbf{q}_t^{\top}\mathbf{kv}_t$（式 5）虽呈线性复杂度，但 "it is inherently unparallelizable"。分块（块长 $m$）后，块内用左积、块间用右积：$\mathbf{O}_1=\mathbf{Q}_1\mathbf{KV}_0+[(\mathbf{Q}_1\mathbf{K}_1^{\top})\odot\mathbf{M}]\mathbf{V}_1$（式 7），$\mathbf{O}_2=\mathbf{Q}_2\mathbf{KV}_1+[(\mathbf{Q}_2\mathbf{K}_2^{\top})\odot\mathbf{M}]\mathbf{V}_2$、$\mathbf{KV}_1=\mathbf{KV}_0+\mathbf{K}_1^{\top}\mathbf{V}_1$（式 8–9），最终复杂度 $O(nd^2+nBd)$，其中 $B$ 为块长（§2.2.1）[^src-minimax-2025-01]。

### 数据流与维度变换

**维度符号**：$n$ 序列长度；$d$ 头维；$B$ 块长；$T=n/B$ 块数；$\mathbf{Q}_t,\mathbf{K}_t,\mathbf{V}_t\in\mathbb{R}^{B\times d}$ 为第 $t$ 块；$\mathbf{q}_t,\mathbf{k}_t,\mathbf{v}_t$ 为单 token 列向量。Algorithm 1 每轮从 HBM 装载 $\mathbf{Q}_t,\mathbf{K}_t,\mathbf{V}_t$ 到片上 SRAM，on-chip 计算后写回（§2.2.1）[^src-minimax-2025-01]。

### Stage 1 · 右积递推（无分块的理想形式）

$$\begin{aligned}
\mathbf{kv}_t &= \mathbf{kv}_{t-1}+\mathbf{k}_t\mathbf{v}_t^{\top} && \text{// } [d,d]+[d,1]\times[1,d]\to[d,d]\ \text{——式(5)：注意力记忆逐 token 累加外积，常数时间更新}\\
\mathbf{o}_t^{\top} &= \mathbf{q}_t^{\top}\mathbf{kv}_t && \text{// } [1,\bcancel{d}]\times[\bcancel{d},d]\to[1,d]\ \text{——读记忆得输出；但递推链跨 } t \text{ 串行，无法并行}
\end{aligned}$$

### Stage 2 · 块内左积（因果部分，SRAM 上并行）

$$\begin{aligned}
\mathbf{O}_{\text{intra}} &= [(\mathbf{Q}_t\mathbf{K}_t^{\top})\odot\mathbf{M}]\mathbf{V}_t && \text{// } [B,\bcancel{d}]\times[\bcancel{d},B]\to[B,B]\xrightarrow{\ \odot M\ \text{块内因果掩码}}\ [B,\bcancel{B}]\times[\bcancel{B},d]\to[B,d]\ \text{——块内保留精确因果注意力，规模受 } B \text{ 约束}
\end{aligned}$$

### Stage 3 · 块间右积与记忆更新

$$\begin{aligned}
\mathbf{O}_{\text{inter}} &= \mathbf{Q}_t\,\mathbf{KV} && \text{// } [B,\bcancel{d}]\times[\bcancel{d},d]\to[B,d]\ \text{——块内全部 query 共享进入本块前的历史记忆}\\
\mathbf{KV} &\leftarrow \mathbf{KV}+\mathbf{K}_t^{\top}\mathbf{V}_t && \text{// } [d,\bcancel{B}]\times[\bcancel{B},d]\to[d,d]\ \text{——处理完本块后把外积累入记忆，供下一块读}
\end{aligned}$$

输出为两段之和 $\mathbf{O}_t=\mathbf{O}_{\text{intra}}+\mathbf{O}_{\text{inter}}$（Algorithm 1）[^src-minimax-2025-01]。与 [[linear-attention-rnn-form]] 对照，式（5）即该页 Stage 3 的 $s_i$ 递推。lightning attention 的贡献不在数学（结合律骨架相同）而在工程：分块把串行的逐 token 递推换成可并行的块间矩阵乘。

## 7:1 混合配置：softmax 层补检索

**为什么需要 softmax 层**。小规模实验发现 lightning attention "demonstrates limited retrieval capabilities"（§2.2.2）[^src-minimax-2025-01]。Figure 7 下游评测中 lightning attention 仅在 NIAH 明显落后（"linear attention exhibits similar language modeling capabilities to Transformer models but falls short in retrieval tasks, rendering it unsuitable for LLMs"，§2.2.2.3）[^src-minimax-2025-01]。§2.2.4（"Going Through a Book"）分析原因：将 softmax 注意力写成逐项递推（式 10–11）后，等价于每步从序列开头重算隐藏状态的线性 RNN，以此保留输入信息；纯线性模型缺少该重算，导致 "hinders their ability to effectively retain input data"（式 12 对照）[^src-minimax-2025-01]。同节给出容量论证：softmax 注意力的检索容量为 $O(d)$，lightning attention 为 $O(d^2/h)$，$d>h$ 时后者更大，因此 hybrid-lightning 的检索与外推能力优于 softmax-only（§2.2.4）[^src-minimax-2025-01]。

**为什么是每 8 层 1 个 softmax**。论文固定「每 8 层 1 个 softmax」（即 7:1）做对照实验链，而非比例扫描：

**Table 3（1B）** 对比 hybrid-lightning 与 hybrid-cosformer2 / hybrid-hgrn2，NIAH 为 95.7 对 43.6 / 91.8，SCR 为 13.3 对 10.9 / 10.8，TGS 33.4K 为三者最高（§2.2.3）[^src-minimax-2025-01]。**Table 4（1B/3B，等速条件）** 对照 hybrid-window（SWA 窗口 256–1024，每 8 层换全 softmax），随窗口加大训练变慢，为对齐速度只取 ≤1024；hybrid-lightning 的 NIAH 在 1B 为 95.7、3B 为 98.0，hybrid-window 全部 ≤57.9，"the hybrid-lightning model outperforms all other models across all metrics"（§2.2.3）[^src-minimax-2025-01]。**Table 5（28B 总参 / 5B 激活 MoE，各训 1T token）** 每 8 连续层把前 7 层的 softmax 换成 lightning attention，BBH 28.2→32.2、DROP 27.4→29.0、MATH 4.6→6.8，除 CMMLU 47.3→46.0 与 GSM8k 18.8→18.5 两处回退外，"improves accuracy across most benchmarks"（§2.3）[^src-minimax-2025-01]。

论文未给出 7:1 与其他比例的消融；1/8 softmax 基于检索缺陷观察直接固定，最终模型 80 层即 10 组 × 7+1（§2）[^src-minimax-2025-01]。§7 将保留 1/8 softmax 列为待解决的妥协："We are investigating more efficient architectures that can eliminate softmax attention entirely"[^src-minimax-2025-01]。

### 7:1 下的注意力成本账

FLOPs 公式出自 Table 1（沿用 Kaplan et al. 2020 口径，排除 embedding 与 subleading 项；$l$ 层数、$b$ batch、$n$ 序列长、$d$ 头维、$h$ 头数）（§2.2.2）[^src-minimax-2025-01]：

### Stage 4 · 三种架构的注意力 FLOPs 与 7:1 的账

$$\begin{aligned}
F_{\text{softmax}} &= 72bnld^2\Big(1+\frac{n}{6d}+\frac{5}{18d}\Big) && \text{// Table 1：长序列主导项 } \tfrac{n}{6d}\text{，随 } n \text{ 线性增长}\\
F_{\text{lightning}} &= 72bnld^2\Big(1+\frac{1}{2h}+\frac{5}{18d}\Big) && \text{// Table 1：无 } n \text{ 相关项，注意力 FLOPs 与序列长解耦}\\
F_{\text{hybrid}} &= 72bnld^2\Big(1+\frac{n}{48d}+\frac{7}{16h}+\frac{5}{18d}\Big) && \text{// Table 1：} \tfrac{n}{48d} \text{ 来自 } \tfrac{1}{8} \text{ softmax 层，} \tfrac{7}{16h} \text{ 来自 } \tfrac{7}{8} \text{ lightning 层}
\end{aligned}$$

公式对应：$\frac{n}{48d}=\frac{1}{8}\cdot\frac{n}{6d}$——每 8 层仅 1 个 softmax 层，序列相关项缩为 softmax-only 的 $1/8$；$\frac{7}{16h}=\frac{7}{8}\cdot\frac{1}{2h}$——7/8 个 lightning 层贡献的序列无关项（$5/18d$ 为两架构公共项，不随注意力类型缩放）[INFERENCE：系数对应关系为本页按 Table 1 三行公式的代数比对]。长序列极限下 $(n/6d)/(n/48d)=8$，7:1 配置把序列相关的注意力 FLOPs 压到 softmax-only 的 $1/8$；代入 MiniMax-01 实配 $d=128$、$h=64$、$n=1{,}024{,}000$ 得 $n/6d\approx1333$、$n/48d\approx167$、$7/16h\approx0.007$，hybrid 的注意力 FLOPs 约为 softmax-only 的 $1/7.96$，且 lightning 层份额完全可忽略 [INFERENCE：数值代入为本页计算]。

## 训练与推理特性

**训练**。70M–7B、至多 300B token 的 scaling law 实验（softmax 配 FlashAttention-2 对照）拟合出 $L(C)=\beta_X C^{\alpha_X}$：同预算下 hybrid 用更多参数与 token 但 loss 更低（Table 2：Hybrid-lightning $L(C)=3.4797C^{-0.0763}$ 对 Softmax $3.7087C^{-0.0798}$）（§2.2.2.2）[^src-minimax-2025-01]。速度上，3B 模型测 TGS（tokens per GPU per second，对照含 HGRN2、Mamba2），lightning attention "achieves a constant training speed irrespective of the sequence length and is the sole linear model that outperforms FlashAttention2"（§2.2.2.4，Figure 8 序列长 1,024–65,536；各曲线数值在 PDF 文本层不可读）[^src-minimax-2025-01]。分布式：长上下文序列并行 LASP+ 把原 LASP 的 send-recv 顺序依赖改为「局部前缀和 + AllGather 同步」，计算速度达原 LASP 的 $1/N_{pcn}$（$N_{pcn}$ 为并行计算节点数，代价是总通信量增加）；varlen ring attention 把 ring attention 直接作用于 data-packing 后的整条序列，避免 padding 浪费（§3.2）[^src-minimax-2025-01]。

**推理**。解码期右积递推仅需维护 $[d,d]$ 的 KV 记忆且每步开销恒定，区别于每步重读全历史 KV 的 softmax 注意力，对应 [[linear-attention-rnn-form]] 的 RNN 读法（§2.2）[^src-minimax-2025-01]。工程上四项 kernel 优化（batched kernel fusion、prefill/decode 分离、multi-level padding、StridedBatchedMatmul 扩展）使 H20 端到端 MFU 超 75%；decode kernel fusion 使解码与短文本端到端延迟降低 10%；prefill/decode 分离使 batch 内长短混合请求延迟从约 100 ms 降至 50 ms；在 1,024,000 token 序列下，softmax attention 注意力延迟占比达 95%，lightning attention 仅占 <12%（§3.3）[^src-minimax-2025-01]。

## 证据：MiniMax-Text-01 的大规模落地

最终模型含 456B 总参、45.9B 激活，设 32 专家 top-2、80 层（7:1）、64 头 × 头维 128、softmax 层 GQA group 8、RoPE 半维 base 10,000 与 hidden 6144（§2）[^src-minimax-2025-01]。训练上下文分三阶段扩至 1M（128K→512K→1M，Table 6），vanilla NIAH 压力测试外推至 4M（Figure 14，"despite only being trained on contexts up to 1M tokens"）（§4.2）[^src-minimax-2025-01]。评测（§5.7）：

通用评测（Table 8，0-shot CoT）中，MMLU 88.5（与 DeepSeek-V3 持平、低于 Llama-3.1-405B 的 88.6），C-SimpleQA 67.4 全场最高，IFEval 89.1，Arena-Hard 89.1；GPQA Diamond 54.4 超 GPT-4o (11-20) 的 46.0；MATH 77.4 超 GPT-4o 76.6 与 Claude-3.5-Sonnet 74.1；编程偏弱，MBPP+ 71.7 为八模型最低，HumanEval 86.9 仅高于 Gemini-1.5-Pro 与 Qwen2.5-72B 的 86.6[^src-minimax-2025-01]。长上下文检索（Table 9，RULER）中，64k 处 0.943 与领先模型差距极小，128k 起全面领先（0.947 对 Gemini-1.5-Pro 0.917），1M 处 0.910 对 Gemini-1.5-Pro 0.850（其余对照在 128k/256k 后无数据）[^src-minimax-2025-01]。长上下文理解（Table 10，LongBench-V2）w/ CoT overall 56.5 对 GPT-4o 51.4 与 Claude-3.5-Sonnet 46.7（Human 基线 53.7）[^src-minimax-2025-01]。长上下文学（Table 11，MTOB eng→kalam）无上下文 6.0 ChrF（全场最低），全书上下文 51.60、Δ+45.6 超所有对照模型[^src-minimax-2025-01]。

## 边界

- **纯线性注意力不适合 LLM**：检索能力是固有缺陷（§2.2.2、§2.2.4 原句见上），7:1 的 softmax 层是补丁，论文自述目标是彻底去掉 softmax（§7）[^src-minimax-2025-01]。
- **7:1 无比例消融**：对照实验链固定每 8 层 1 个 softmax，未扫其他比例；「为什么恰好 7」只有 Table 3–5 的间接支持[^src-minimax-2025-01]。
- **质量代价出现在细节上**：Table 5 中 CMMLU、GSM8k 两项回退；Table 8 中编程基准垫底，论文归因于预训练编程数据有限（§7）[^src-minimax-2025-01]。
- **长上下文评测偏人工场景**：§7 自述 "Current evaluation datasets for long-context retrieval tasks are primarily designed for artificial or simplified scenarios"；NIAH 在训练早期（128K steps 内）即达峰，不能反映后续训练收益（§4.2）[^src-minimax-2025-01]。
- **部署成本**：456B 总参的规模由「单机 8×80GB、8-bit 量化下处理 1M token」的约束反推（§1、§2.4）；"affordable cost" 是论文自述口径[^src-minimax-2025-01]。
- **推导为简化版**：lightning attention 推导省略 normalization、SiLU 与 gating（§2.2.1）；Eq. 9 原文求和号内写作 $\mathbf{k}_m\mathbf{v}_m^{\top}$，系论文自身笔误[^src-minimax-2025-01]。

## 名称消歧

DeepSeek-V3.2 DSA 中的 lightning indexer 是为稀疏注意力逐 token 打分的独立 FP8 打分网络，主注意力仍是 softmax（§2.3）[^src-2512-02556-deepseek-v3-2]；其仅名称相似，与本页 lightning attention（线性注意力的 IO-aware 实现）无关，详见 [[deepseek-sparse-attention]]。

## 在本 wiki 中的位置

[[linear-attention-rnn-form]] 给出 lightning attention 复用的结合律骨架与右积递推的原始出处；[[flash-attention]] 将同一 IO-aware 思想应用于精确 softmax 注意力（tiling 减 HBM 访问），lightning attention 则把打分数学本身线性化、彻底消去 $N^2$ 项；[[native-sparse-attention]] 是另一条减注意力成本的路线——不改打分数学、改依赖集合（训练期块稀疏选择）。[[source-minimax-2025-01]] 为源摘要。

[^src-minimax-2025-01]: [[source-minimax-2025-01]]
[^src-2512-02556-deepseek-v3-2]: [[source-2512-02556-deepseek-v3-2]]
