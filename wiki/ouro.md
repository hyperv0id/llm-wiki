---
title: "Ouro（Looped Language Model）"
type: technique
tags:
  - architecture
  - weight-tying
  - latent-reasoning
  - test-time-compute
  - adaptive-computation
  - language-model
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Ouro（Looped Language Model）

Ouro 是字节跳动 Seed 等机构提出的 Looped Language Model（LoopLM）模型族：同一组 $L$ 层 Transformer 栈在前向中循环运行 $t$ 轮，token 仍按因果顺序生成，额外的串行计算发生在单个 token 的隐状态上[^src-seed-2025-ouro]。Ouro-1.4B 与 Ouro-2.6B 采用 7.7T token 预训练、默认循环 4 轮；两者在多数基准上对齐 4B/8B dense 模型，摘要称「match the results of up to 12B SOTA LLMs」[^src-seed-2025-ouro]。名字取自递归的 Ouroboros。

## 问题与定位

论文将 LLM 扩容归纳为两条既有路径：扩大参数量与训练数据，或在推理期用 Chain-of-Thought 生成更多 token；Ouro 走第三条路，在固定参数预算内通过权重共享的循环加深内部计算图（§1）[^src-seed-2025-ouro]。该方案带来三点收益：学习式早停让简单输入少迭代、复杂输入多迭代；加深内部计算图而不拉长输出序列；同数据下每个参数的容量更高（§1）[^src-seed-2025-ouro]。论文针对循环在万亿 token 级是否仍有超出参数缩放增益的问题，给出了肯定回答（§1）[^src-seed-2025-ouro]。

论文将 Universal Transformer 列为该路线源头，并给出两种理解（§2）：一是参数共享，将循环视作 ALBERT 式的跨深度复用；二是隐空间推理，将每轮迭代视作不落成文字的「thought」以逐步修订表示[^src-seed-2025-ouro]。二者在论文中并列呈现，互不替代。

## 结构与配置

对 $L$ 层栈 $\mathcal{M}_L=T_{\theta_L}\circ\cdots\circ T_{\theta_1}$，循环模型定义为

$$F^{(t)}(\cdot)=\mathrm{lmhead}\circ\underbrace{\mathcal{M}_L\circ\cdots\circ\mathcal{M}_L}_{t\ \text{次}}\circ\,\mathrm{emb}(\cdot),\qquad F^{(1)}\equiv F.$$

原文写作 $T_\theta^{L}\circ\cdots\circ T_\theta^{1}$，上标为层号；栈内 $L$ 层各有独立参数，整栈在 $t$ 轮间共享，因此 $t=1$ 时即为普通 $L$ 层 Transformer（式 1，§3.1）[^src-seed-2025-ouro]。

| 模型 | 参数量 | 层数 $L$ | 隐藏维 $d$ | 注意力 | FFN | 位置编码 | 词表 |
|---|---|---|---|---|---|---|---|
| Ouro-1.4B | 1.4B | 24 | 2048 | MHA | SwiGLU | RoPE | 49,152 |
| Ouro-2.6B | 2.6B | 48 | 2048 | MHA | SwiGLU | RoPE | 49,152 |

两模型共用 SmolLM2 的 49,152 词表 tokenizer；每个子层前均设 RMSNorm，论文称为 sandwich normalization（Table 2、§4.1）[^src-seed-2025-ouro]。2.6B 并非从零训练，而是将 1.4B 的 24 层复制为 48 层再做后续阶段；循环结构的跨轮共享使这种 upcycling 比标准 Transformer 更平滑（§4.3.1）[^src-seed-2025-ouro]。

没有单独的 prelude 与 coda：输入嵌入直接进入循环栈，读出由每轮 LM head 完成（Figure 3、§3.1）[^src-seed-2025-ouro]。每个循环步同时接 LM head 与退出门 $\lambda_t=\sigma(\mathrm{Linear}_\phi(h^{(t)}))$；门参数 $\phi$ 在各轮间共享（§3.2）[^src-seed-2025-ouro]。

## 数据流与维度

**维度符号**：$M$ 序列长度（token 数）；$d=2048$ 隐藏维；$L\in\{24,48\}$ 栈内层数；$T$ 循环轮数，生产模型 $T_{\max}=4$；$n_h$ 注意力头数（论文只写 MHA，未给数值）；$d_h=d/n_h$ 单头通道维；$d_{\mathrm{ff}}$ SwiGLU 中间维（论文未给数值）；$V=49152$ 词表；$b$ 每元素字节数。形状记作 $[\text{行},\text{列}]$。

### Stage 1 · 嵌入 token

$$\begin{aligned}
h^{(0)} &= \mathrm{emb}(x) && \text{// } [M]\xrightarrow{\text{逐位置查表}}[M,d]\ \text{——每个 token 得到一行 } d \text{ 维向量，作为第 1 轮循环的输入}
\end{aligned}$$

### Stage 2 · 第 $t$ 轮穿过同一组 $L$ 层

$$\begin{aligned}
\tilde h &= h+\mathrm{Attn}\!\left(\mathrm{RMSNorm}(h)\right)W_O && \text{// } [M,\bcancel{d}]\times[\bcancel{d},d]\to[M,d]\ \text{——注意力子层写回残差流，} W_O \text{ 把头输出拼回 } d \text{ 维}\\
T_{\theta_\ell}(h) &= \tilde h+\mathrm{FFN}\!\left(\mathrm{RMSNorm}(\tilde h)\right) && \text{// } [M,d]\to[M,d]\ \text{——SwiGLU FFN 逐位置作用，不在位置之间交换信息}\\
h^{(t)} &= \left(T_{\theta_L}\circ T_{\theta_{L-1}}\circ\cdots\circ T_{\theta_1}\right)\!\left(h^{(t-1)}\right) && \text{// } [M,d]\to[M,d]\ \text{——同一组 } L \text{ 层权重被第 } t \text{ 轮再次调用，形状不变，材料化深度累加为 } tL
\end{aligned}$$

注意力按 MHA 展开，RoPE 加在 $Q,K$ 上，causal mask 后沿 key 轴计算 softmax：

$$\begin{aligned}
Q,K,V &= \mathrm{RMSNorm}(h)W_Q,\ \mathrm{RMSNorm}(h)W_K,\ \mathrm{RMSNorm}(h)W_V && \text{// } [M,\bcancel{d}]\times[\bcancel{d},d]\to[M,d]\ \text{——三个投影各自输出 } n_h \text{ 个头，每头 } d_h=d/n_h\\
S &= QK^{\top}/\sqrt{d_h} && \text{// } [n_h,M,\bcancel{d_h}]\times[n_h,\bcancel{d_h},M]\to[n_h,M,M]\ \text{——消去头通道维，得到每个头对全部 key 的打分}\\
\mathrm{Attn} &= \mathrm{softmax}(S)V && \text{// } [n_h,M,\bcancel{M}]\times[n_h,\bcancel{M},d_h]\to[n_h,M,d_h]\ \text{——沿被 mask 的 key 轴加权求和，拼头后回 } [M,d]
\end{aligned}$$

循环不加宽 2.6B，而是复用同一栈：$T=4$ 时 1.4B 材料化 96 层、2.6B 材料化 192 层，参数量仍是 $L$ 层那一份[^src-seed-2025-ouro]。

### Stage 3 · 每个循环步插入 LM head 与退出门

$$\begin{aligned}
o^{(t)} &= \mathrm{lmhead}\!\left(h^{(t)}\right) && \text{// } [M,\bcancel{d}]\times[\bcancel{d},V]\to[M,V]\ \text{——每轮都读出词表分布，第 } t \text{ 步的 NLL 从这里算}\\
\lambda^{(t)} &= \sigma\!\left(\mathrm{Linear}_\phi\!\left(h^{(t)}\right)\right) && \text{// } [M,d]\times[d,1]\to[M,1]\ \text{——逐位置瞬时退出概率，} \phi \text{ 在所有轮之间共享}
\end{aligned}$$

Ouro 的中间监督不设在最后一轮：每个循环步都接同一个 LM head，损失按退出分布对各步 NLL 加权（式 2、4，§3.1、§3.3）[^src-seed-2025-ouro]。

### Stage 4 · 由瞬时退出概率构造退出分布

$$\begin{aligned}
S_t &= \prod_{j=1}^{t}\left(1-\lambda_j\right),\quad S_0\equiv 1 && \text{// } [M]\ \text{——逐位置生存概率，即前 } t \text{ 轮都没退出的概率}\\
\tilde p_t &= \lambda_t S_{t-1} && \text{// } [M]\ \text{——恰在第 } t \text{ 轮第一次退出的未归一化概率}\\
p_\phi(t\mid x) &= \tilde p_t\ (t<T_{\max}),\quad p_\phi(T_{\max}\mid x)=S_{T_{\max}-1} && \text{// } [M]\ \text{——剩余质量并入末轮，使逐位置 } \sum_t p_\phi=1\\
\mathrm{CDF}(m\mid x) &= \sum_{t=1}^{m}p_\phi(t\mid x) && \text{// } [M]\to[M]\ \text{——逐位置累积分布；推理取第一个 } \mathrm{CDF}(m)\ge q \text{ 的 } m \text{ 退出}
\end{aligned}$$

推理采用 Q-exit：给定阈值 $q$，在首个累积退出概率超过 $q$ 的轮次停止；$q$ 小则早停、$q$ 大则更深（§3.2，引 PALBERT）[^src-seed-2025-ouro]。

### Stage 5 · 训练损失：退出分布加权各步 NLL + 熵正则

$$\begin{aligned}
\ell_i^{(t)} &= -\log p_\theta^{(t)}\!\left(x_{i+1}\mid x_{1:i}\right) && \text{// } [M-1]\ \text{——第 } t \text{ 轮在第 } i \text{ 个位置的 next-token NLL}\\
\mathcal{L} &= \mathbb{E}_{x}\sum_{i=1}^{M-1}\left[\sum_{t=1}^{T_{\max}}p_\phi(t\mid x_{1:i})\,\ell_i^{(t)}-\beta H\!\left(p_\phi(\cdot\mid x_{1:i})\right)\right] && \text{// } [M-1]\times[T_{\max}]\xrightarrow{\text{按 }p_\phi\text{ 加权求和}}[M-1]\ \text{——每位置先对轮次取期望，再对位置求和}
\end{aligned}$$

第一项是各步 NLL 的凸组合，第二项将退出分布拉向均匀先验。均匀先验下 $\mathrm{KL}(p_\phi\|\pi)=-H(p_\phi)+\log T_{\max}$，式 4 与 $\beta$-加权变分目标仅差一个与参数无关的常数，$\beta=1$ 时即负 ELBO；$\beta$ 控制探索与利用（§3.3）[^src-seed-2025-ouro]。Ouro 的 $\beta$ 从 Stage 1a 的 0.1 降至 Stage 1b 起的 0.05（Table 1）[^src-seed-2025-ouro]。

### Stage 6 · Stage II：冻结 LM，只训退出门

$$\begin{aligned}
I_i^{(t)} &= \max\!\left(0,\ L_{i,\mathrm{stop}}^{(t-1)}-L_{i,\mathrm{stop}}^{(t)}\right) && \text{// } [M]\ \text{——第 } t \text{ 轮相对上一轮的损失下降；} L_{\mathrm{stop}} \text{ 已 detach，不回传到 LM}\\
w_i^{(t)} &= \sigma\!\left(50\left(I_i^{(t)}-0.005\right)\right) && \text{// } [M]\ \text{——理想续跑概率，下降超过 0.005 时接近 1，停滞时接近 0}\\
\mathcal{L}_{\mathrm{adaptive}} &= \frac{1}{T_{\max}}\sum_{t=2}^{T_{\max}}\left[-\frac{1}{M}\sum_{i=1}^{M}\left(w_i^{(t)}\log(1-\lambda_i^{(t)})+(1-w_i^{(t)})\log\lambda_i^{(t)}\right)\right] && \text{// } [M]\xrightarrow{\text{逐位置 BCE 后按 }M\text{ 平均}}\text{标量}\ \text{——门预测的续跑概率对理想标签做 BCE，只更新 } \phi
\end{aligned}$$

式 6 同时惩罚两类错误：该续跑却退出（underthinking）、该退出却续跑（overthinking）（§3.4）[^src-seed-2025-ouro]。Stage II 冻结 LM 参数单独训练，只调整深度分配，不改 token 级预测（§3.2）[^src-seed-2025-ouro]。

### Stage 7 · 推理期 loop 数与算力记账

$$\begin{aligned}
C_{\text{tok}} &\approx T_{\mathrm{exit}}\cdot L\cdot\left(4Md+6d\,d_{\mathrm{ff}}\right) && \text{// 标量 FLOPs [INFERENCE]}\ \text{——} T_{\mathrm{exit}} \text{ 由 CDF 分位决定；参数只存 } L \text{ 层，材料化深度为 } T_{\mathrm{exit}}L\\
\mathrm{KV}_{\text{naive}} &= T\cdot L\cdot 2\cdot M\cdot d\cdot b && \text{// bytes}\ \text{——每轮每层各存一份 K/V；} b \text{ 为每元素字节数}\\
\mathrm{KV}_{\text{last}} &= L\cdot 2\cdot M\cdot d\cdot b && \text{// bytes}\ \text{——解码期只留末轮 KV，内存降 } T \text{ 倍（Table 14）}
\end{aligned}$$

式中略去 QKV/O 投影与 LM head，二者仅在 $M\gg d$ 时为低阶项；$d_{\mathrm{ff}}$ 论文未给数值。算力随 $T_{\mathrm{exit}}$ 线性增长，参数量不随之增长；注意力项随 $M$ 线性增长，FFN 项与 $M$ 无关。$q$ 是部署时控制算力—精度折中（compute–accuracy trade-off）的旋钮（§3.2、§5.4.1）[^src-seed-2025-ouro]。

## 训练配方

预训练分四段数据，合计约 7.7T token；门训练分 Stage I（与预训练联合）与 Stage II（冻结 LM 单独训练）（Table 1、§4.3.1）[^src-seed-2025-ouro]。

| 阶段 | 训练 token | 循环步 | 序列长 | 学习率 | $\beta$ | 数据重点 |
|---|---|---|---|---|---|---|
| Stage 1a Pre-train I | 3T | 8 | 4K | $3\times10^{-4}$ 常数 | 0.1 | Web 高、Math/Code 低 |
| Stage 1b Pre-train II | 3T | 4 | 4K | $3\times10^{-4}$ 常数 | 0.05 | Web 高、Math/Code 低 |
| Stage 2 CT Annealing | 1.4T | 4 | 16K | $3\times10^{-5}$ 余弦 | 0.05 | Web 中、Math/Code 高 |
| Stage 3 LongCT | 20B | 4 | 64K | $3\times10^{-5}$ 常数 | 0.05 | 长上下文 |
| Stage 4 Mid-training | 300B | 4 | 32K | $1\times10^{-5}$ 余弦 | 0.05 | 高质量 SFT 混合 |

优化器统一为 AdamW：weight decay 0.1、gradient norm clip 1.0、$\beta_1=0.9$、$\beta_2=0.95$（Table 1）[^src-seed-2025-ouro]。

Stage 1a 采用 8 轮，出现 loss spike 与梯度振荡后降至 4 轮；batch 从 4M token 增至 8M；循环结构比参数匹配的 Transformer 需要更小的学习率，但未做系统 LR sweep（§4.3）[^src-seed-2025-ouro]。Stage 4 的 300B 由 90B 新采样的 SFT 混合与 Stage 1 的 30B、Stage 2 的 180B replay 组成（§4.2）[^src-seed-2025-ouro]。

SFT 用 8.3M 条样本（Math 3.5M、Code 3.2M、Science 808K、Chat 767K），以 2 个 epoch、32K 序列、Adam、学习率 $2\times10^{-5}$ 得到 Ouro-Thinking（§4.4、Table 6）[^src-seed-2025-ouro]。后续 DAPO/GRPO 尝试未超过 SFT checkpoint：vLLM/SGLang 的固定执行路径不支持可变深度，off-policy rollout 与固定 4 轮 RL 均未带来增益（§4.5）[^src-seed-2025-ouro]。

## 证据

### 基座模型（Table 7、Table 8）

| 模型（参数 / 训练 token） | MMLU | MMLU-Pro | BBH | ARC-C | HellaSwag | Winogrande | GSM8K | MATH500 | HumanEval | HumanEval+ | MBPP | MBPP+ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ouro-1.4B R4（1.4B / 7.7T） | 67.35 | 48.62 | 71.02 | 60.92 | 74.29 | 72.30 | 78.92 | 82.40 | 74.40 | 67.40 | 73.00 | 62.70 |
| Qwen3-4B（4.0B / 36T） | 73.19 | 51.40 | 70.95 | 63.65 | 75.66 | 71.19 | 72.86 | 59.60 | 77.40 | 70.70 | 78.80 | 65.90 |
| Gemma3-4B（4.0B / 4T） | 58.37 | 34.61 | 66.32 | 60.92 | 75.58 | 71.07 | 68.69 | 68.60 | 34.80 | 29.30 | 60.60 | 51.10 |
| Ouro-2.6B R4（2.6B / 7.7T） | 74.60 | 55.73 | 80.46 | 66.40 | 79.69 | 75.85 | 81.58 | 90.85 | 78.70 | 70.70 | 80.40 | 66.60 |
| Qwen3-8B（8.0B / 36T） | 76.63 | 53.72 | 77.65 | 66.10 | 79.60 | 76.80 | 83.09 | 62.30 | 84.80 | 75.30 | 79.00 | 67.90 |
| Gemma3-12B（12.0B / 12T） | 72.14 | 49.21 | 78.41 | 72.44 | 83.68 | 77.74 | 77.18 | 83.20 | 46.30 | 37.20 | 73.50 | 66.10 |

1.4B 在 BBH（71.02 vs 70.95）、GSM8K（78.92 vs 72.86）、MATH500（82.40 vs 59.60）上不低于 Qwen3-4B，在 MMLU、MMLU-Pro、ARC-C、HellaSwag、HumanEval、HumanEval+、MBPP、MBPP+ 上低于它；论文表述为「comparable to the 4B Qwen3-Base across most benchmarks」（§5.1）[^src-seed-2025-ouro]。2.6B 在 MMLU-Pro（55.73 vs 53.72）、BBH（80.46 vs 77.65）、MATH500（90.85 vs 62.30）、MBPP（80.40 vs 79.00）上高于 Qwen3-8B，在 MMLU、Winogrande、GSM8K、HumanEval、HumanEval+、MBPP+ 上低于它；论文总结为「outperforms dense models up to 8B on reasoning-intensive benchmarks」（§5.1）[^src-seed-2025-ouro]。摘要另写「match the results of up to 12B SOTA LLMs」，结论段又写「match or exceed 4B and 8B standard transformers」，三处口径不完全一致[^src-seed-2025-ouro]。

训练 token 数未对齐：Ouro 7.7T，Qwen3-4B/8B 36T，Gemma3-4B 4T、12B 12T（Table 7–8）[^src-seed-2025-ouro]。评测设置见 Table 16：MMLU 5-shot logprobs、MMLU-Pro 5-shot CoT、BBH 3-shot CoT、GSM8K 3-shot CoT、MATH500 5-shot 自建管线，代码使用 evalplus pass@1[^src-seed-2025-ouro]。

### 推理模型（Table 9）

| 模型 | AIME24 pass@1 / pass@10 | AIME25 pass@1 / pass@10 | OlympiadBench | BeyondAIME | HLE | SuperGPQA | GPQA |
|---|---|---|---|---|---|---|---|
| Ouro-1.4B-Thinking-R4 | 65.0 / 83.3 | 46.3 / 73.3 | 71.6 | 34.0 | 5.21 | 47.4 | 45.5 |
| Ouro-2.6B-Thinking-R4 | 64.7 / 90.0 | 50.3 / 76.7 | 76.4 | 39.0 | 5.58 | 53.7 | 52.7 |
| Qwen3-4B | 61.3 / 75.0 | 51.3 / 63.3 | 73.2 | 31.0 | 5.21 | 51.9 | 54.5 |
| Qwen3-8B | 73.0 / 86.7 | 66.7 / 81.3 | 75.3 | 38.0 | 2.22 | 48.0 | 59.1 |

1.4B 的 AIME24 高于 Qwen3-4B（65.0 vs 61.3），AIME25 低于它（46.3 vs 51.3）；2.6B 的 OlympiadBench 与 BeyondAIME 略高于 Qwen3-8B（76.4 vs 75.3、39.0 vs 38.0），AIME24/25 与 GPQA 低于它（64.7 vs 73.0、50.3 vs 66.7、52.7 vs 59.1）[^src-seed-2025-ouro]。评测基于同一 in-house harness 与 LLM-as-judge，temperature 1.0、top_p 0.7（Table 17）[^src-seed-2025-ouro]。

### 循环步数与外推（Table 10–13）

基座模型在训练深度 $T=4$ 达到峰值，外推至 $T=5$–$8$ 性能下降（Table 10、11）[^src-seed-2025-ouro]：

| 模型 / 步 | ARC-C | ARC-E | C-QA | HellaSwag | MMLU | Winogrande |
|---|---|---|---|---|---|---|
| 1.4B, T=1 | 37.63 | 63.85 | 44.64 | 55.24 | 41.21 | 56.99 |
| 1.4B, T=4 | 60.92 | 83.96 | 75.43 | 74.29 | 67.45 | 72.30 |
| 1.4B, T=8 | 58.19 | 82.07 | 73.55 | 71.60 | 64.49 | 69.30 |
| 2.6B, T=1 | 47.95 | 72.39 | 57.58 | 68.94 | 51.55 | 61.48 |
| 2.6B, T=4 | 66.38 | 86.95 | 81.65 | 79.56 | 74.60 | 75.53 |
| 2.6B, T=8 | 64.76 | 86.49 | 81.08 | 79.50 | 72.24 | 74.59 |

Thinking 模型在 $T=1$ 几乎无法解题，1.4B 的 AIME24 为 0.00、OlympiadBench 为 2.22；1.4B 峰值在 $T=4$ 或 $T=5$，2.6B 峰值在 $T=3$ 或 $T=4$（Table 12、13）[^src-seed-2025-ouro]。

| 模型 / 步 | OlympiadBench | SuperGPQA | AIME24 | AIME25 |
|---|---|---|---|---|
| 1.4B Thinking, T=1 | 2.22 | 2.03 | 0.00 | 0.33 |
| 1.4B Thinking, T=4 | 71.55 | 47.37 | 65.00 | 46.30 |
| 1.4B Thinking, T=5 | 72.30 | 48.73 | 60.67 | 47.00 |
| 1.4B Thinking, T=8 | 66.81 | 42.88 | 38.67 | 38.00 |
| 2.6B Thinking, T=1 | 18.96 | 15.66 | 3.00 | 2.00 |
| 2.6B Thinking, T=3 | 75.56 | 56.70 | 70.33 | 50.67 |
| 2.6B Thinking, T=4 | 76.44 | 53.68 | 64.70 | 50.30 |
| 2.6B Thinking, T=8 | 39.26 | 46.84 | 39.00 | 24.33 |

安全对齐是外推退化的例外：$T$ 从 1 增至 8，HEx-PHI 的 harmfulness score 与 harmful rate 反而改善，即使 $T>4$ 没见过（§5.3、§7.1、Figure 8）[^src-seed-2025-ouro]。

### 早停与 KV 缓存（Figure 5、Table 14）

静态退出从 1 轮到 2 轮将 MMLU 从约 40% 提至约 60%，3 轮到 4 轮仅达 67.35%（Table 10 的 MMLU 在 $T=4$ 记 67.45，与 67.35 并存）；学到的门在平均 2.5 轮时约 66%，标准熵正则门约 64%，隐状态差阈值与门相差 1–2 个点，专门训练的 Stage II 门在各算力点高 2–3 个点（§5.4.1）[^src-seed-2025-ouro]。

Table 14 在 Ouro-1.4B 基座上比较三种解码期复用方式：

| 解码期 KV 策略 | GSM8K | MATH-500 | 内存 |
|---|---|---|---|
| Full（4 份 cache） | 78.92 | 82.40 | 1.00× |
| 只留第 1 轮 | 18.73 | 8.43 | 4.00× |
| 只留第 4 轮 | 78.85 | 80.40 | 4.00× |
| 四轮平均 | 78.73 | 78.52 | 4.00× |

prefill 阶段四轮各需一份 KV，复用会掉 10 个点以上（GSM8K）；decode 阶段只留末轮即可把内存降至 $1/4$，GSM8K 只差 0.07，MATH-500 差 2.00（Table 14、§5.4.2）[^src-seed-2025-ouro]。

### 先验与稳定性消融（Appendix A、§4.3）

论文在 776M LoopLM、$T_{\max}=4$、FineWeb-Edu 20B token 上对比了几何先验 $\eta\in\{0.1,\dots,0.9\}$ 与均匀先验：均匀先验的训练损失更低、收敛更干净，几何先验随 $\eta$ 增大平台更高、后期振荡更大（Figure 10、§A）[^src-seed-2025-ouro]。固定平均步数下，均匀先验的 accuracy–compute Pareto 前沿更好；$\eta$ 较大时先验把质量压在 $t=1,2$，深层几乎拿不到训练信号（§A）[^src-seed-2025-ouro]。为抑制 loss spike 与梯度振荡，训练改用 8 轮→4 轮、batch 4M→8M、$\beta$ 0.1→0.05（§4.3）[^src-seed-2025-ouro]。

### 机制：知识容量不涨，知识操作变好（§6）

Capo 合成传记任务上（1M–40M 参数、$N=20$K–$500$K、1000 exposures），同参数量的 loop-1 与 loop-4 模型知识容量均在约 2 bits/parameter，循环未提高容量或容量 scaling，论文据此认为参数量本身是知识容量的直接指标（§6.1、Figure 6 左）[^src-seed-2025-ouro]。Mano 模 23 算术树上，同参数量对照一致偏向循环模型；同 FLOPs 的 $12\otimes1$ 对照上，$6\otimes2$ 循环模型只在长度 24 略高（35.1 vs 34.8），长度 10/16 略低（93.4 vs 93.6、88.5 vs 94.4）（Figure 6 右内嵌表；表头长度是最大表达式长度，不是层数）[^src-seed-2025-ouro]：

| 模型（层数 $\otimes$ 循环数） | 长度 10 | 长度 16 | 长度 24 |
|---|---|---|---|
| Base（12 $\otimes$ 1） | 93.6 | 94.4 | 34.8 |
| Base（2 $\otimes$ 1） | 21.5 | 8.4 | 7.5 |
| Loop（2 $\otimes$ 6） | 98.1 | 96.3 | 78.0 |
| Base（3 $\otimes$ 1） | 75.4 | 29.8 | 11.0 |
| Loop（3 $\otimes$ 4） | 97.9 | 95.8 | 92.2 |
| Base（6 $\otimes$ 1） | 84.7 | 59.5 | 20.0 |
| Loop（6 $\otimes$ 2） | 93.4 | 88.5 | 35.1 |

在 3-hop 合成多跳 QA（120K 唯一 QA 对、20K 步、batch 2048）中，循环模型用更少唯一 QA 对即实现完全泛化，同 token 预算下学得更快；对照基线包含同参数（6 $\otimes$ 1）与同 FLOPs（24 $\otimes$ 1）（§6.2、Figure 7）[^src-seed-2025-ouro]。MMLU 57 个子类的逐类增益走向一致：elementary mathematics +155.6%、formal logic +143.3%、logical fallacies +127.8%、high school statistics +126.9%，而 moral scenarios +7.8%、global facts +8.3%、virology +13.7%、anatomy +21.4%（Table 15、§B.4）[^src-seed-2025-ouro]。

理论给出受限构造：在邻接列输入、全注意力与逐元素布尔阈值下，一层单头、隐宽 $2n$ 的循环 Transformer 用 $O(1+\log_2\max\{1,D\})$ 轮解出含参数内知识图的可达性查询；相同假设下离散 CoT 为 $O(n^2)$、连续 CoT 为 $O(1+D)$、Universal Transformer 为 $O(1+\log\max\{1,D\})$（Theorem 1、§B.5）[^src-seed-2025-ouro]。该界衡量串行轮数而非总计算量：每轮在全宽 $2n$ 上的稠密矩阵乘为 $O(n^3)$；构造未覆盖 causal mask 与标准归一化[^src-seed-2025-ouro]。

### 安全与忠实性（§7、Figure 8–9）

HEx-PHI 含 330 条、11 个禁止类别，GPT-4o 打 1–5 分，harmful rate 取 5 分比例；Ouro-Thinking 在 4 轮时 harmful rate 为 1.4B 的 0.009、2.6B 的 0.003，Qwen3-4B-Thinking 为 0.009（§7.1）[^src-seed-2025-ouro]。基座模型在 1–8 轮上的 harmfulness score 随轮数下降，5–8 轮属于训练深度之外的外推（Figure 8a）[^src-seed-2025-ouro]。

忠实性以 Quora Question Pairs 的 1000 对问题为观察性代理：在单个循环步内，第 $i$ 步答案能被第 $24i$ 层的线性探针预测，但上一层 $24(i-1)$ 探针无法预测，表明新一轮在改变答案；相邻轮一致数 A[2,3]=551、A[2,4]=361，$i\ge4$ 时 A[i,i+1] 接近 1000，训练只到 4 轮，之后答案收敛到不动点（§7.2、Figure 9）[^src-seed-2025-ouro]。对照 Qwen3-4B-Thinking 的末 token 线性探针 ROC AUC 为 0.99，即答案在思维链前已基本确定[^src-seed-2025-ouro]。该实验未做干预，无法像 CoT 那样测平均处理效应，中间文本只是隐状态的读数而非机制本身（§7.2）[^src-seed-2025-ouro]。

### Scaling law（Appendix D）

小规模实验用 53M、134M、374M、778M、1.36B 五个规模，$T\in\{1,2,4,8\}$，每个点 20B token 的 FineWeb-Edu。总损失拟合为

$$L_t=E+\frac{A}{(N+t_1)^\alpha}+\frac{B}{(D+t_2)^\beta}+\frac{C}{(T_m+t_3)^\gamma},$$

全点拟合 $R^2=0.9596$；留出规模、数据、最大轮数的泛化 $R^2$ 分别为 0.9542、0.9385/0.9609/0.962、0.9581（§D.2、§E.1）[^src-seed-2025-ouro]。分步损失把 $T_m$ 换成当前轮 $T$：

$$L_s=E+\frac{A}{(N+t_1)^\alpha}+\frac{B}{(D+t_2)^\beta}+\frac{C}{(T+t_3)^\gamma},$$

$T_m=2/4/8$ 的全点 $R^2$ 为 0.8898/0.8146/0.795；分步损失的泛化略低（规模 0.8815/0.797，数据 0.8686/0.8882/0.8896 与 0.793/0.813/0.8142，轮数 0.8118）（§D.2、§E.2）[^src-seed-2025-ouro]。$\gamma>0$ 表示分步损失随轮数下降；论文把总损失写成退出分布加权的分步损失加熵项，在 $T_m=4$ 时用学到的分布 $\{0.0004,0.0855,0.3793,0.5348\}$ 回代，重建总损失的 $R^2=0.961$（§D.3、Figure 19–20）[^src-seed-2025-ouro]。

同参数、同数据对照下，标准 Transformer 的平均基准分高于 LoopLM，差距随循环轮数增加、总体上随模型规模增大而减小（Table 18）[^src-seed-2025-ouro]：

| 模型规模 | Step 2 | Step 4 |
|---|---|---|
| 170M | 0.021 | 0.039 |
| 340M | 0.023 | 0.037 |
| 680M | 0.015 | 0.026 |
| 1.3B | 0.017 | 0.025 |

表注定义差距为「Standard − LoopLM」。Table 18 的规模标签是 170M / 340M / 680M / 1.3B，与 §D.1 的五档（53M / 134M / 374M / 778M / 1.36B）对不上，论文未解释两套标签的对应关系。这组实验的 token 预算 20B、规模 $\leq 1.36$B 均远小于 7.7T 主实验，论文未把两者接成同一条曲线[^src-seed-2025-ouro]。论文另报告一个反常现象：模型规模不足时，浅层分步损失会随数据量上升；作者归因于门把更多权重移向深层以压低总损失，并建议 LoopLM 采用更大模型规模（§D.3）[^src-seed-2025-ouro]。

## 边界（论文自述与页面读出的限制）

只在 $T=4$ 训练；$T=5$–$8$ 的外推在基座与 Thinking 基准上普遍退化，安全是例外（Table 10–13、Figure 8）[^src-seed-2025-ouro]。同参数小规模对照中标准 Transformer 仍占优，差距 0.015–0.039；主实验 7.7T 结果缺少同 token、同算力的 dense 对照，参数效率结论依赖跨模型基准比较[^src-seed-2025-ouro]。1.4B/2.6B 与 4B/8B/12B 训练 token 数从 4T 到 36T 不等，评测管线也不完全相同（基座用 lm-eval/evalplus，Thinking 用 in-house LLM-as-judge）（Table 7–9、16–17）[^src-seed-2025-ouro]。RLVR 尝试中 off-policy rollout 与固定 4 轮均未超过 SFT；原因归于可变深度与现有推理引擎固定执行路径不兼容，以及小模型经大量 SFT 后提升空间有限（§4.5）[^src-seed-2025-ouro]。KV 复用在 prefill 不成立，复用会掉 10 个点以上；decode 仅留末轮虽省 4× 内存，MATH-500 仍从 82.40 降至 80.40（Table 14）[^src-seed-2025-ouro]。图可达性定理假设局限于邻接列输入、全注意力、布尔阈值与隐宽 $2n$，未覆盖 causal mask 与标准归一化，且每轮 $O(n^3)$ 工作，轮数对数界不等于总算力对数界（§B.5）[^src-seed-2025-ouro]。忠实性分析属无干预的观察性代理；Stage I/II 目标均不保证每多一轮 next-token 损失必降，深度收益须按任务与深度实测（§7.2、§7.3）[^src-seed-2025-ouro]。2.6B 由 24 层复制成 48 层，并非从零训练的 48 层对照；Stage 1 含中文数据但 tokenizer 无中文词表，Stage 2 起移除中文（§4.2、§4.3.1）[^src-seed-2025-ouro]。

## 谱系

论文将 Universal Transformer 视为源头，并在 §2 从两种互补视角梳理该脉络：参数共享（ALBERT、Takase 的共享策略、Megrez2 的跨层专家复用）与隐空间推理（Coconut 的 continuous thought token、CoTFormer 把激活写回输入、Saunshi 的 looped transformer、Huginn 的 recurrent depth）[^src-seed-2025-ouro]。论文明确区分 explicit feedback 与 implicit LoopLM：Coconut/CoTFormer 将隐状态或激活重新喂回输入，Ouro 的「thought」仅存在于轮次间的隐状态演化中（§2）[^src-seed-2025-ouro]。

论文贡献涵盖 7.7T token 规模下的参数效率，以及均匀先验熵正则的深度分配（Stage I 与 Stage II）；KV 复用与安全/忠实性分析在正文另行列出（§1、§3.3、§3.4、§5.4.2、§7）[^src-seed-2025-ouro]。停机机制采纳 PALBERT 的 Q-exit 与 PonderNet 的变分视角（§3.2、§3.3），未讨论其与 Universal Transformer 中 ACT 停机的差异[^src-seed-2025-ouro]。论文未提供与 Universal Transformer 或 Huginn 的数值对照，仅在相关工作作定位；两者的细节见 [[universal-transformer]] 与 [[recurrent-depth]]。

## 相关页面

[[recurrent-depth]] 同属循环深度路线，记录 Huginn 的 prelude/core/coda 结构、截断反传与推理期策略；[[universal-transformer]] 是同一组权重沿深度施加 $T$ 步与 ACT 停机的源头；[[latent-reasoning]] 讨论隐空间推理的问题设定与可监督性；[[source-seed-2025-ouro]] 为本源摘要。

[^src-seed-2025-ouro]: [[source-seed-2025-ouro]]
