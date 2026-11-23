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
last_updated: 2026-11-22
source_count: 1
confidence: medium
status: active
---

# Ouro（Looped Language Model）

堆参数与堆数据都要求「更多」，而同一组层反复穿过几次，可以在参数不变的前提下多花算力——这条自 Universal Transformer 起就存在的路线一直没被推到万亿 token 规模。Ouro 是字节跳动 Seed 等机构提出的 LoopLM 模型族：token 仍按因果顺序生成，额外的串行计算发生在单个 token 的隐状态上，简单输入可由学到的门提前退出。Ouro-1.4B 与 Ouro-2.6B 用 7.7T token 预训练、默认循环 4 轮，作者报告二者在多数基准上对齐 4B/8B dense 模型，摘要称可 match 到 12B 级 SOTA；名字取自咬住自己尾巴的 Ouroboros[^src-seed-2025-ouro]。

## 问题与定位

Ouro 被定位为「第三条路」：不靠更多参数或更长的思维链，而是在固定参数预算内用权重共享的循环加深内部计算图。由此换来三点——学习式早停让简单输入少迭代、复杂输入多迭代；加深的是内部计算图而不是输出序列长度；同数据下每个参数承担更多计算。论文要回答的核心问题是循环在万亿 token 级是否仍有超出参数缩放的增益，其答案为肯定，增益集中在推理类任务（§1）[^src-seed-2025-ouro]。

论文把 Universal Transformer 列为源头，并给出两种互补理解：参数共享（把循环视作 ALBERT 式的跨深度复用）与隐空间推理（把每轮迭代视作不落成文字的「thought」，逐步修订表示）。二者并列，互不替代（§2）[^src-seed-2025-ouro]。

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

Stage 1a 用 8 轮训练出现 loss spike 与梯度振荡，降到 4 轮；batch 从 4M token 增至 8M；循环结构比参数匹配的 Transformer 需要更小的学习率，但未做系统 LR sweep。Stage 4 的 300B 由 90B 新采样 SFT 混合加 Stage 1 的 30B、Stage 2 的 180B replay 组成（§4.2、§4.3）[^src-seed-2025-ouro]。

论文在 776M 规模上比对几何先验与均匀先验，后者训练损失更低、收敛更干净，$\eta$ 大时深层几乎拿不到训练信号，故统一用均匀先验的熵正则（Appendix A）[^src-seed-2025-ouro]。SFT 用 8.3M 条样本（Math 3.5M、Code 3.2M、Science 808K、Chat 767K），2 个 epoch、32K 序列、学习率 $2\times10^{-5}$，得到 Ouro-Thinking；后续 DAPO/GRPO 未超过 SFT checkpoint，vLLM/SGLang 的固定执行路径不支持可变深度（§4.4、§4.5、Table 6）[^src-seed-2025-ouro]。

## 证据

1. **定位：以约 1/5 的训练 token 追上大 2–3 倍的 dense 模型，胜在推理密集基准，不是全面领先。** Ouro-1.4B（1.4B / 7.7T）对 Qwen3-4B（4.0B / 36T）：BBH 71.02 对 70.95、GSM8K 78.92 对 72.86、MATH500 82.40 对 59.60 领先，MMLU 67.35 对 73.19、MMLU-Pro 48.62 对 51.40、ARC-C 60.92 对 63.65 与代码三项落后。Ouro-2.6B 对 Qwen3-8B 是同一格局：MMLU-Pro 55.73 对 53.72、BBH 80.46 对 77.65、MATH500 90.85 对 62.30 领先，MMLU 74.60 对 76.63、GSM8K 81.58 对 83.09、HumanEval 78.70 对 84.80 落后；Thinking 版互有胜负（1.4B 的 AIME24 65.0 对 61.3、AIME25 46.3 对 51.3）。token 预算本就不对齐，Qwen3 用 36T、Gemma3-4B 只用 4T（Table 7–9）[^src-seed-2025-ouro]。
2. **归因：循环轮数是主要增益来源，也是外推天花板；它改的是知识操作而非知识容量。** 基座模型在训练深度 $T=4$ 达到峰值，$T=8$ 回落（1.4B MMLU 67.45→64.49、ARC-C 60.92→58.19；2.6B MMLU 74.60→72.24）；Thinking 模型 $T=1$ 几乎不能解题（1.4B AIME24 0.00、OlympiadBench 2.22），1.4B 峰值在 $T=4$–$5$、2.6B 在 $T=3$–$4$，$T=8$ 的 OlympiadBench 只剩 66.81 与 39.26；早停门本身有效：静态退出从 1 轮到 2 轮把 MMLU 从约 40% 提到约 60%，学到的门在平均 2.5 轮时约 66%。安全对齐是唯一例外，$T$ 从 1 增到 8 反而改善。合成任务给出机制解释：Capo 传记任务上同参数量的 loop-1 与 loop-4 知识容量都约 2 bits/parameter，循环没提高容量；Mano 模 23 算术树上同参数量一致偏向循环模型，同 FLOPs 的 $12\otimes1$ 对照里 $6\otimes2$ 只在长度 24 略高（35.1 对 34.8）；MMLU 57 个子类的增益集中在数学与形式逻辑（elementary mathematics +155.6%、formal logic +143.3%），moral scenarios 与 global facts 仅 +7.8% / +8.3%（Table 10–13、15、Figure 5/8、§6）[^src-seed-2025-ouro]。
3. **缺口与负结果：同参数、同数据下标准 Transformer 的基准均值更高。** 差距随循环轮数增加而扩大（Step 2 → Step 4）、随规模增大而缩小，$T=4$ 档从 170M 的 0.039 降到 1.3B 的 0.025（Table 18）；该表的 170M/340M/680M/1.3B 规模档与 §D.1 的 53M/134M/374M/778M/1.36B 对不上，论文未解释对应关系，论文另给总损失与分步损失的对数缩放律拟合（全点 $R^2$ 0.96 与 0.80–0.89），同样只跑 20B token、至多 1.36B 规模，未与 7.7T 主实验接成同一条曲线。主实验缺同 token、同算力的 dense 对照；KV 复用在 prefill 不成立（GSM8K 掉 10 点以上），decode 只留末轮省 4× 内存但 MATH-500 从 82.40 降到 80.40；忠实性分析靠无干预的线性探针，只能作观察性代理（Table 14、§D、§7.2）[^src-seed-2025-ouro]。

## 边界（论文自述与页面读出的限制）

只在 $T=4$ 训练；$T=5$–$8$ 的外推在基座与 Thinking 基准上普遍退化，安全是例外（Table 10–13、Figure 8）[^src-seed-2025-ouro]。同参数小规模对照中标准 Transformer 仍占优，差距 0.015–0.039；主实验 7.7T 结果缺少同 token、同算力的 dense 对照，参数效率结论依赖跨模型基准比较[^src-seed-2025-ouro]。1.4B/2.6B 与 4B/8B/12B 训练 token 数从 4T 到 36T 不等，评测管线也不完全相同（基座用 lm-eval/evalplus，Thinking 用 in-house LLM-as-judge）（Table 7–9、16–17）[^src-seed-2025-ouro]。RLVR 尝试中 off-policy rollout 与固定 4 轮均未超过 SFT；原因归于可变深度与现有推理引擎固定执行路径不兼容，以及小模型经大量 SFT 后提升空间有限（§4.5）[^src-seed-2025-ouro]。KV 复用在 prefill 不成立，复用会掉 10 个点以上；decode 仅留末轮虽省 4× 内存，MATH-500 仍从 82.40 降至 80.40（Table 14）[^src-seed-2025-ouro]。图可达性定理假设局限于邻接列输入、全注意力、布尔阈值与隐宽 $2n$，未覆盖 causal mask 与标准归一化，且每轮 $O(n^3)$ 工作，轮数对数界不等于总算力对数界（§B.5）[^src-seed-2025-ouro]。忠实性分析属无干预的观察性代理；Stage I/II 目标均不保证每多一轮 next-token 损失必降，深度收益须按任务与深度实测（§7.2、§7.3）[^src-seed-2025-ouro]。2.6B 由 24 层复制成 48 层，并非从零训练的 48 层对照；Stage 1 含中文数据但 tokenizer 无中文词表，Stage 2 起移除中文（§4.2、§4.3.1）[^src-seed-2025-ouro]。

## 谱系

论文将 Universal Transformer 视为源头，从两种互补视角梳理该脉络：参数共享（ALBERT、Takase 的共享策略、Megrez2 的跨层专家复用）与隐空间推理（Coconut 的 continuous thought token、CoTFormer 把激活写回输入、Saunshi 的 looped transformer、Huginn 的 recurrent depth）（§2）[^src-seed-2025-ouro]。论文明确区分 explicit feedback 与 implicit LoopLM：Coconut/CoTFormer 将隐状态或激活重新喂回输入，Ouro 的「thought」仅存在于轮次间的隐状态演化中（§2）[^src-seed-2025-ouro]。

论文贡献涵盖 7.7T token 规模下的参数效率，以及均匀先验熵正则的深度分配（Stage I 与 Stage II）；KV 复用与安全/忠实性分析在正文另行列出（§1、§3.3、§3.4、§5.4.2、§7）[^src-seed-2025-ouro]。停机机制采纳 PALBERT 的 Q-exit 与 PonderNet 的变分视角，未讨论其与 Universal Transformer 中 ACT 停机的差异（§3.2、§3.3）[^src-seed-2025-ouro]。论文未提供与 Universal Transformer 或 Huginn 的数值对照，仅在相关工作作定位；两者的细节见 [[universal-transformer]] 与 [[recurrent-depth]]。

## 相关页面

[[recurrent-depth]] 同属循环深度路线，记录 Huginn 的 prelude/core/coda 结构、截断反传与推理期策略；[[universal-transformer]] 是同一组权重沿深度施加 $T$ 步与 ACT 停机的源头；[[latent-reasoning]] 讨论隐空间推理的问题设定与可监督性；[[source-seed-2025-ouro]] 为本源摘要。

[^src-seed-2025-ouro]: [[source-seed-2025-ouro]]
