---
title: "Recurrent Depth（循环深度 / Looped Transformer）"
type: technique
tags:
  - architecture
  - test-time-compute
  - weight-tying
  - latent-reasoning
  - language-model
created: 2026-10-25
last_updated: 2026-10-27
source_count: 4
confidence: medium
status: active
---

# Recurrent Depth

Recurrent depth（又称 recurrent-depth transformer、looped transformer）把一个共享权重的层块在**深度方向**反复施加，而不是在时间步上递归：固定深度 Transformer 的层数是训练时定死的，循环深度把一部分层换成单一块 `R`，前向时迭代 $r$ 次，于是串行计算量由推理时的 $r$ 决定、权重只存一份[^src-2502-05171-recurrent-depth][^src-tosea-looped-transformer]。Huginn（arXiv:2502.05171）是这条路线目前公开实验中规模最大的一次：3.5B 参数、800B token，训练时平均循环 32 次[^src-2502-05171-recurrent-depth]。

## 三条 scaling 轴

论文把循环深度定位为与已有两条轴并列的第三条（§1）：预训练缩放参数量，Chain-of-Thought 缩放生成的 token 数，循环深度缩放每个 token 的串行迭代次数；作者自述三者不冲突[^src-2502-05171-recurrent-depth]。作者的动机措辞是「把昂贵的内部推理强制投影成一个可 verbalize 的 next token 显得浪费」[^src-2502-05171-recurrent-depth]。二手报道把同一主张写成「compute 变成运行时的旋钮」与「推理不必落成文字」两句话，并把它与 [[mixture-of-experts|MoE]] 的镜像关系放在一起讲[^src-tosea-looped-transformer]。

## 结构：prelude / core / coda

模型拆成三段（§3.1）：prelude `P` 把 token 嵌入隐空间；core `R` 是唯一的循环单元，迭代 $r$ 次；coda `C` 解出下一 token 的概率[^src-2502-05171-recurrent-depth]。主模型形状 $(l_P,l_R,l_C)=(2,4,2)$、$h=5280$、55 个 96 维头、MLP 内层 17920，参数按 prelude+head 1.5B、core 1.5B、tied embedding 0.5B 分，共 3.5B；展开深度 $2+4r+2$，$r=32$ 时 132 层[^src-2502-05171-recurrent-depth]。循环单元是**整个 4 层的块**，不是逐层做权重共享[^src-2502-05171-recurrent-depth]。

四个设计决定带明确理由：

| 决定 | 论文给出的理由 |
|---|---|
| 每轮重新注入输入嵌入 `e`（拼接后经 $A:\mathbb{R}^{2h}\to\mathbb{R}^h$），不靠加法 | 只给 $s_0$ 时迭代解只依赖边界条件、不稳定；`R` 不依赖 `e` 就不可能是单调算子，无法表示对数据依赖的严格凸函数的梯度下降。小模型上加法同样可行，大规模拼接更好[^src-2502-05171-recurrent-depth] |
| 没有 step embedding（迭代步号不进模型） | 试过 $s_i=R_i(e,s_{i-1})$，与 path independence 冲突，模型无法外推[^src-2502-05171-recurrent-depth] |
| $s_0$ 从截断正态采样而非置零 | 与每步注入合起来使迭代收敛到与初值无关的不动点，即 path independence[^src-2502-05171-recurrent-depth] |
| sandwich norm（attention 与 MLP 前后各一道归一化） | 小尺度下 pre-norm/post-norm 都可用，大尺度上是训练 recurrence 的必要条件：一次失败的预训练在换成 sandwich norm 后才恢复[^src-2502-05171-recurrent-depth] |

输出只在最后一轮：中间迭代没有 token 级监督，损失是最后一轮的标准 next-token 交叉熵对 $r$ 的期望[^src-2502-05171-recurrent-depth]。论文没有用扩散式加噪，自述初步实验无帮助（§3.1 Remark 3.1）[^src-2502-05171-recurrent-depth]。

## 数据流与维度

形状记作 $[\text{行}, \text{列}]$。**维度符号**：$n$ 为序列长度；$h=5280$ 为隐藏维；$H=55$ 为头数、$d=96$ 为单头通道维（$Hd=h$）；$l_P=2$、$l_R=4$、$l_C=2$ 为三段层数；$r$ 为循环次数；$k=8$ 为反向传播穿过的迭代数；$|V|=65536$。论文按单头书写，本页的多头展开与 `n_c` 位置标 [INFERENCE]。

### Stage 1 · 嵌入输入（prelude）

$$\begin{aligned}
e &= P_\theta(x) && \text{// } [n]\xrightarrow{\gamma E(x)\ \text{后过 } l_P=2 \text{ 层}}[n, h]\ \text{——} \mathtt{E}\in\mathbb{R}^{|V|\times h} \text{ 输出乘 } \sqrt{h}\text{，} e \text{ 在每一轮循环里被重新读入}
\end{aligned}$$

### Stage 2 · 初始化循环状态

$$\begin{aligned}
s_0 &\sim \operatorname{TruncNorm}\!\left(0,\ \sigma_s^2 I_{n\cdot h}\right),\quad \sigma_s^2=2/5 && \text{// } [n, h]\ \text{——初值与输入无关，只提供迭代起点；收敛点由 } e \text{ 与 } R \text{ 决定}
\end{aligned}$$

### Stage 3 · 循环块第 $i$ 轮（$i=1,\dots,r$）

$$\begin{aligned}
u_i^{(0)} &= A\,[\,s_{i-1};\,e\,] && \text{// } [n, \bcancel{2h}]\times[\bcancel{2h}, h]\to[n, h]\ \text{——消去拼接维，把上一轮状态与本轮输入压回 } h \text{ 维}\\
Q,K,V &= u_i^{(\ell-1)}W_Q,\ u_i^{(\ell-1)}W_K,\ u_i^{(\ell-1)}W_V && \text{// } [n, \bcancel{h}]\times[\bcancel{h}, Hd]\to[H, n, d]\ \text{——} Q,K \text{ 过 RoPE（base 50000），共 } l_R=4 \text{ 层}
\end{aligned}$$

$$\begin{aligned}
S &= KQ^{\top}/\sqrt{d} && \text{// } [H, n, \bcancel{d}]\times[H, \bcancel{d}, n]\to[H, n, n]\ \text{——消去 } d\text{，causal 掩码后 softmax 沿 key 轴}\\
u_i^{(\ell)} &= n_2\!\left(u_i^{(\ell-1)}+\operatorname{Attn}(n_1(u_i^{(\ell-1)}))W_O\right) && \text{// } [H, n, d]\xrightarrow{\text{拼头}}[n, Hd]=[n, h]\ \text{——} W_O \text{ 压回 } h \text{ 维，sandwich 归一化 } n_1,n_2\\
s_i &= n_c\!\left(u_i^{(l_R)}\right) && \text{// } [n, h]\to[n, h]\ \text{——块末再归一化，作为下一轮或 coda 的输入}
\end{aligned}$$

四层共用同一组权重，$s_i$ 的形状与 $s_{i-1}$ 相同，所以 $r$ 可以任意取；每多一轮相当于多 4 层材料化深度，$r=32$ 时总深度 132 层[^src-2502-05171-recurrent-depth]。

### Stage 4 · 解码（coda）

$$\begin{aligned}
p &= \operatorname{softmax}\!\left(C_\theta(s_r)\,E^{\top}\right) && \text{// } [n, \bcancel{h}]\times[\bcancel{h}, |V|]\to[n, |V|]\ \text{——} C \text{ 为 } l_C=2 \text{ 层加 } n_c\text{，输出头与输入 embedding 绑定}
\end{aligned}$$

### Stage 5 · 训练路径：随机深度 + 截断反传

$$\begin{aligned}
\tau &\sim \mathcal{N}\!\left(\log\bar r-\tfrac12\sigma^2,\ \sigma\right),\quad r\sim\mathcal{P}(e^{\tau})+1 && \text{// 标量，} \bar r=32,\ \sigma=\tfrac12\ \text{——log-normal Poisson，重尾；实测均值 33.0、中位 29.0、众数 24.0}\\
\mathcal{L} &= \mathbb{E}_{x}\,\mathbb{E}_{r\sim\Lambda}\ \operatorname{CE}\!\left(m_\theta(x,r),\ x'\right) && \text{// } r \text{ 在损失内取期望，同一个 } r \text{ 贯穿整条前向}
\end{aligned}$$

反向只穿过最后 $k=8$ 轮（深度方向的截断 BPTT），实现上再叠加逐迭代的 gradient checkpointing，于是激活显存与反向计算量与 $r$ 无关；prelude 因为每轮被注入而每步都拿得到梯度[^src-2502-05171-recurrent-depth]。训练时同一 micro-batch 内的 $r$ 被锁成同一个值（locked-step sampling），作者自认这与式中的期望有偏差[^src-2502-05171-recurrent-depth]。

### Stage 6 · 推理路径：把 $r$ 当旋钮

推理期全部 zero-shot，不加训练[^src-2502-05171-recurrent-depth]：

- 直接扫 $r\in\{1,4,8,16,32,64\}$，有任务相关饱和点（§5.3）。
- 逐 token 早停：相邻两步的 $\mathrm{KL}(s_i,s_{i-1})<5\times10^{-4}$ 即停止并采样该 token；没有训练 exit head。
- 早停导致 KV 不等长时，直接 attend 到 cache 中最新、最深的那份 KV，而不是重算。
- KV 预算 $s$：第 $i$ 步读写槽位 $i \bmod s$，把 cache 压到固定大小。
- warm start（continuous CoT）：用上一 token 的 $s_r$ 初始化本 token 的 $s_0$，跨 token 串起更长的计算图，收敛步数平均少 1–2 步。
- self-speculative decoding：少迭代起草、多迭代验证，起草期的状态在验证时复用。

## 证据

主结果是 3.5B/0.8T 的 Huginn-0125 与 1B–12B 的开源固定深度 Transformer 对比（lm-eval-harness，zero-shot，normalized accuracy）[^src-2502-05171-recurrent-depth]：

| $r$ | ARC-E | ARC-C | HellaSwag | MMLU | OBQA | PiQA | SciQ | WinoGrande | GSM8K CoT |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 49.07 | 27.99 | 43.46 | 23.39 | 28.20 | 64.96 | 80.00 | 55.24 | — |
| 8 | 65.11 | 35.15 | 58.54 | 25.29 | 35.40 | 73.45 | 92.10 | 55.64 | — |
| 16 | 69.49 | 37.71 | 64.67 | 31.25 | 37.60 | 75.79 | 93.90 | 57.77 | — |
| 32 | 69.91 | 38.23 | 65.21 | 31.38 | 38.80 | 76.22 | 93.50 | 59.43 | 34.80/42.08 |

对照：OLMo-2-1124-7B 在 ARC-E/ARC-C/HellaSwag/MMLU 上是 82.79/57.42/80.50/60.56（Table 1）[^src-2502-05171-recurrent-depth]。

三条读数：

1. **推理能力几乎全部来自循环**。同一架构只过 core 一次（$r=1$）时，0.8T checkpoint 的 GSM8K CoT 是 0.00、ARC-C 24.06；180B 的早期 checkpoint 从 $r=1$ 到 $r=32$，GSM8K CoT 由 0.00 升到 9.02，而固定深度基线在同数据同设置下只有 1.82（Table 4）[^src-2502-05171-recurrent-depth]。二手报道把这条读成「架构本身而非训练配方带来差距」[^src-tosea-looped-transformer]。
2. **饱和点与任务强相关，且外推不退化**。HellaSwag 约 8 次到峰值；ARC-C 的 few-shot 饱和点随示例数从 8–12 次（0-shot）推到约 32 次（25–50-shot）；GSM8K CoT 与 HumanEval 继续吃到更多循环（Fig 7–9）[^src-2502-05171-recurrent-depth]。
3. **等效参数量是 FLOP 口径**。论文原话是预训练消耗的 FLOPs「接近一个 32B 参数 transformer」，测试时能到「FLOP 预算等价于标准 50B 固定深度 transformer」，摘要写作「计算负载等价于 50B 参数」[^src-2502-05171-recurrent-depth]。精度上它与 OLMo-7B 大致相当、落后于 OLMo-2，不是 50B 级模型[^src-2502-05171-recurrent-depth]。Fig 1 横轴的「Materialized Parameters 3.6B…103B」是按展开层数折算的材料化参数量，不是精度等效曲线[^src-2502-05171-recurrent-depth]。

## 训练与工程

最终模型 800B token（schedule 排到 795B）、47000 步，全局 batch 16M token，序列长 4096，BPE 65536；非循环基线 180B token[^src-2502-05171-recurrent-depth]。硬件是 Oak Ridge Frontier 的 512 节点（4096 张 MI250X），21 个不超过 12 小时的 segment，bf16，只做 DDP + 节点内 ZeRO-1 + 逐迭代 checkpointing，没有 TP/PP；规模化后 52–64 TFLOP/s per GPU（41–51% AFU）、1–1.2M token/s[^src-2502-05171-recurrent-depth]。数据配比偏 code（25.36%）与 math（6.14%），作者明确说没有做配比消融[^src-2502-05171-recurrent-depth]。

训练有明确的失败模式：一次 run 的隐状态在 token 维相关性升到 1（表示坍缩）；另一次 run 恢复了相关性但学成忽略 $s$，$r=1$ 与 $r=32$ 的验证困惑度相同，即陷入无法利用 test-time compute 的局部极小；最终 run 靠恢复 sandwich norm 与降低峰值学习率修好[^src-2502-05171-recurrent-depth]。学习率在两处的数值不一致：§4.1 写 $\eta=5\times10^{-4}$，§4.3 写最终 run 的峰值 $4\times10^{-5}$，两值并列、未择一[^src-2502-05171-recurrent-depth]。

## 边界

- 论文自述仍是 proof-of-concept：只训练了一个大模型，数据混合未验证，学习率从未 cooldown（§5、§10）[^src-2502-05171-recurrent-depth]。
- 架构消融基本是定性的：输入注入、step embedding、拼接 vs 加法、归一化布局、截断长度 $k$ 都只有论证或单条曲线，没有数值对照表；带逐项数字的只有 Table 4（循环 vs 非循环）与 Table 6（推理期方案）[^src-2502-05171-recurrent-depth]。
- 显式 CoT 的收益依赖循环打底：$r=1$ 时 CoT 无效（GSM8K CoT 0.00），$r=32$ 时才有 34.80/42.08；论文没有讨论「latent 推理不如长 CoT」的情形[^src-2502-05171-recurrent-depth]。
- 算术是硬的失败面：2 操作数加法可稳到 4–5 位数，4–6 个操作数的单位数加法几乎做不对（4 operands/digit=1 仅 0.3），且顺序随循环次数不一致；模型未针对算术微调（附录 A.1）[^src-2502-05171-recurrent-depth]。
- 代码落后专用模型：MBPP 24.80 vs StarCoder2-3B 43.00，HumanEval 23.17 vs 31.09（Table 3）[^src-2502-05171-recurrent-depth]。
- 只做了单段循环；多段循环、把 CoT 内化进循环、按难度做 RL、与 MoE/线性注意力结合都列在 future work（§9）[^src-2502-05171-recurrent-depth]。

## 谱系与相邻路线

论文只在相关工作里提谱系：recurrent depth 被反复重新发现（RNN、扩散模型、universal/looped transformer）；Dehghani et al. 用循环深度追求图灵完备[^src-dehghani-2019-universal-transformers]——一手源核实（[[universal-transformer]]，§2、§4）：UT 把同一组自注意力 + transition 权重沿深度施加 $T$ 步、每步重加正弦位置-时间编码，固定 $T$ 即全层权重绑定的多层 Transformer，另配 ACT 逐位置停机，与 Huginn 的块级共享、无停机、每轮重注入输入嵌入不同；Lan et al. 在规模上用过但固定循环次数；直接同源的是「随机 unroll + 每步注入输入」能学可泛化算法的一线工作；与 deep equilibrium model 可视为同族的「学习一个不动点迭代」（§8）[^src-2502-05171-recurrent-depth]。论文没有给出与 Universal Transformer 的逐条对照（未讨论 ACT 式逐位置停机、层间共享与块级共享的差别），也没有引用其 halting 机制[^src-2502-05171-recurrent-depth]。

二手报道给出的时间线（Universal Transformers 2018 → ALBERT 2019 跨层参数共享 → Coconut 2024 连续思维 → Huginn 2025 → Mixture-of-Recursions 2025 逐 token 递归深度 → Ouro/LoopLM 2025 学习退出）只出现在博客里，未在论文正文核实[^src-tosea-looped-transformer]；其中 Universal Transformers 一条现已入库一手源 [[source-dehghani-2019-universal-transformers]]，Ouro 一条已入库 [[source-seed-2025-ouro]]（结构见 [[ouro]]）。Ouro 把 24/48 层整栈循环 4 轮，每轮都接 LM head 与退出门，损失按退出分布加权[^src-seed-2025-ouro]；Huginn 的监督点只在末轮（见本页「结构」段）。

与 MoE 的关系是论文 §9 明确写下的互补：本架构 compute-heavy、材料化参数多于实际参数，MoE 相反；作者原话是循环深度擅长学习推理模式、MoE 擅长存储与检索复杂信息，并设想未来的 recurrent MoE 能在多轮迭代里反复路由到同一 expert[^src-2502-05171-recurrent-depth]。

[^src-2502-05171-recurrent-depth]: [[source-2502-05171-recurrent-depth]]
[^src-tosea-looped-transformer]: [[source-tosea-looped-transformer]]
[^src-dehghani-2019-universal-transformers]: [[source-dehghani-2019-universal-transformers]]
[^src-seed-2025-ouro]: [[source-seed-2025-ouro]]
