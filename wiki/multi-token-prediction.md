---
title: "Multi-token Prediction"
type: technique
tags:
  - llm
  - pretraining
  - training-objective
  - sample-efficiency
  - code-generation
created: 2026-07-27
last_updated: 2026-10-27
source_count: 3
confidence: medium
status: active
---

# Multi-token Prediction

**Multi-token prediction（多 token 预测）** 是一种语言模型预训练目标：在每个位置同时预测后续 **n** 个 token，而非仅预测 next token。Gloeckle et al. (ICML 2024) 给出可扩展至 13B、训练时间与峰值显存几乎无额外开销的实现，并在代码生成与部分自然语言生成任务上证明其样本效率与推理加速收益[^src-gloeckle-2024-multi-token-prediction]。

## 动机

标准 next-token + teacher forcing 容易：

1. 贴合局部、易预测模式，忽视决定后续语义轨迹的 **choice points**；
2. 训练（真值前缀）与推理（自回归前缀）分布错配（exposure bias 的 LM 版本）[^src-gloeckle-2024-multi-token-prediction]。

多 token 预测把“看更远”写进主损失，作为辅助任务强化长期依赖与规划。

## 方法

**共享 trunk + n 个独立 head + 共享 unembedding**。论文把 $n$ 个未来 token 的联合预测按共享表示 $z_{t:1}$ 拆为 $n$ 个逐头交叉熵之和（式 (2)），完整数据流见下[^src-gloeckle-2024-multi-token-prediction]：

为保持参数公平，新增 head 层数从 trunk 层数中扣除，总参数与 next-token 基线相同[^src-gloeckle-2024-multi-token-prediction]。推理默认只用 $i=1$ 头，其余头支撑 [[self-speculative-decoding|self-speculative decoding]]（见「推理侧」小节）。附录探索了线性、causal 堆叠与 anticausal 等备选 head，**并行独立 transformer head** 最稳[^src-gloeckle-2024-multi-token-prediction]。

### 数据流与维度变换

**维度符号**：$T$ 序列长度（token 数）；$d$ trunk 隐藏维；$V$ 词表大小；$n$ 同时预测的未来 token 数。每个 head $f_{h_i}$ 为一层 transformer，共享 unembedding $f_u$[^src-gloeckle-2024-multi-token-prediction]。形状记 $[\text{行}, \text{列}]$。

#### Stage 1 · 共享 trunk 编码上下文

$$\begin{aligned}
z_{t:1} &= f_s(x_{t:1}) && \text{// } [T, d]\ \text{——trunk 只前向一次，表示被 } n \text{ 个 head 复用（Figure 2）}
\end{aligned}$$

#### Stage 2 · n 个独立 head 各自预测 $x_{t+i}$

parallel 架构下各 head 仅读取共享 $z$ 且彼此不交互；输出层共享同一 unembedding（§2、Appendix B）[^src-gloeckle-2024-multi-token-prediction]：

$$\begin{aligned}
h^{(i)} &= f_{h_i}(z_{t:1}) && \text{// } [T, d] \xrightarrow{\text{head } i \text{ 的一层 transformer，逐头独立}} [T, d]\ \text{——无跨头信息流}\\
\ell^{(i)} &= f_u(h^{(i)}) && \text{// } [T, \bcancel{d}]\times[\bcancel{d}, V]\to[T, V]\ \text{——共享 unembedding 消去 } d\\
P_\theta(x_{t+i}\mid x_{t:1}) &= \mathrm{softmax}(\ell^{(i)}) && \text{// } [T, V] \xrightarrow{\text{沿 } V \text{ 维逐行 softmax}} [T, V]\ \text{——第 } i \text{ 步预测分布；} i=1 \text{ 即推理用的 next-token 头}
\end{aligned}$$

#### Stage 3 · 各 head 交叉熵求和

$$\begin{aligned}
L^n &= -\sum_t \sum_{i=1}^{n} \log P_\theta(x_{t+i}\mid z_{t:1})\,P_\theta(z_{t:1}\mid x_{t:1}) && \text{// } [T, V] \xrightarrow{\text{按目标 } x_{t+i} \text{ 取 } -\log\text{，对 } i=1..n \text{ 与位置 } t \text{ 求和}} \text{标量}\ \text{——} n \text{ 个头的损失直接相加，无头间归一化}
\end{aligned}$$

条件独立假设：式 (2) 将联合 $\log P_\theta(x_{t+n:t+1}\mid z_{t:1})$ 展开为 $\sum_{i=1}^n \log P_\theta(x_{t+i}\mid z_{t:1})$，即给定共享 $z$ 时各步预测相互独立；头间自依赖仅在 Appendix B 的 causal / anticausal 变体中引入[^src-gloeckle-2024-multi-token-prediction]。因子 $P_\theta(z_{t:1}\mid x_{t:1})$ 在原文中按式 (2) 原样保留，论文未进一步展开其读法。

### 显存账：峰值 $O(V+d)$ 而非 $O(nV+d)$

朴素实现同时物化 $n$ 个 head 的 logits 及其梯度——每位置各 $(n, V)$——而 $V \gg d$，vocab 向量成为显存瓶颈，压小了可用 batch size（§2）[^src-gloeckle-2024-multi-token-prediction]。论文改为重排 forward/backward 顺序（Figure 2，n=2 图示）[^src-gloeckle-2024-multi-token-prediction]：

1. trunk 前向一次，$z_{t:1}$ 留存；
2. 对 head $i$ 串行执行 forward + backward：该头的 logits 与梯度算完即释放，只把 $d$ 维的 $\partial L^n/\partial f_s$ 累积到 trunk；
3. 进入 head $i{+}1$，重复至 $i=n$。

任一时刻仅存一份 vocab 向量，长期驻留的只有 $d$ 维 trunk 梯度，峰值显存由 $O(nV+d)$ 降至 $O(V+d)$，运行时间无额外开销（§2）[^src-gloeckle-2024-multi-token-prediction]。每个 head 复制一份 unembedding 需要 $(d, nV)$ 的大矩阵，大规模训练不可行（Appendix B）；仅 byte 模型采用 replicated unembeddings（Table S13 表注）[^src-gloeckle-2024-multi-token-prediction]。

> [!note] 运行时口径
> 正文写「at no expense in runtime」，而 Table S5 实测 n=4 相对 next-token 训练为 1.07–1.22×（n=2 为 1.02–1.07×），论文把这点开销归因于逐头 backward 失去 FSDP 通信与计算的 overlap、重新实现可消除；两处口径并列保留[^src-gloeckle-2024-multi-token-prediction]。

### 推理侧：self-speculative decoding 的对照

Gloeckle et al. 的多头模型属于 [[self-speculative-decoding|self-speculative decoding]] 谱系：Stern et al. (2018) 的 blockwise parallel decoding 用**线性头**起草、原模型验证；本文将线性头替换为一层 transformer head，并通过训练侧串行重排使全部 $n$ 个损失项参与训练（Stern 原方案随机选一个头算损失）；且兼容 Medusa 式 tree attention（每头取 top-k 起草）（§3.2、Related work）[^src-gloeckle-2024-multi-token-prediction]。加速上界为所用 head 数 $k$（Table S2 图注）：7B 4-token 模型在 code 达 3.05×（3.50 tokens/forward）、wikipedia 2.74×、books 2.67×，batch size 1–40 大致稳定（Figure S10）；byte 模型 n=8 用 8 头达 6.39×（Table S3）[^src-gloeckle-2024-multi-token-prediction]。多头质量源自多 token **预训练**，区别于事后微调头（§3.2）[^src-gloeckle-2024-multi-token-prediction]。

## 何时有效

| 设定 | 现象 | 出处 |
|------|------|------|
| 模型规模 | 0.3B / 0.6B 全面弱于 baseline；3B 起 n=4 在 MBPP/HumanEval 六项指标首次全面不低于 n=1（如 MBPP pass@1 12.7 对 11.1）；13B 增益最大：MBPP pass@1 30.5 对 26.0、HumanEval pass@1 15.8 对 14.1 | 图 3、表 S7 |
| 最优 n | 32k vocab 代码上 n=4 最稳（HumanEval/MBPP 各 pass@k 领先：MBPP +3.8/+2.1/+3.2，HumanEval +1.2/+3.7/+4.1）；字节级 n=8 更一致；APPS/Intro 可偏 n=6（+0.7/+3.0/+5.3） | §3.4、表 1 |
| 多 epoch | 1T tokens（4 epoch）仍有增益，幅度缩小（MBPP pass@1 +2.4、HumanEval pass@100 +3.2） | §3.5、表 1 |
| 微调 | 多 token **预训练**表征更强；接 next-token 微调（$n{=}4$, $n'{=}1$）往往最好 | §3.6、图 4 |
| 任务类型 | 代码 pass@k、摘要 ROUGE、算法合成任务↑；multiple-choice NL 基准 $\approx$（n=2）或略↓（n=4） | §3.7、图 5/6 |
| 字节模型 | 显著缓解 next-byte 的局部过拟合（MBPP pass@1 +67%、HumanEval pass@1 +20%），byte-level 预训练更可行 | §3.3、表 1 |

表中结果均出自 Gloeckle et al.[^src-gloeckle-2024-multi-token-prediction]，绝对值与差值口径（200 采样估计、温度 oracle）见 §3.3–§3.5 与 Table S12。

## 机制解释（论文自述推测）

论文在「Why does it work? Some speculation」一章中提出三条解释[^src-gloeckle-2024-multi-token-prediction]：

1. **Choice-point 重加权**（§5.1、Appendix L.3）：n-step 预测下，一个 choice point 及其后果共贡献 $1+2+\cdots+n=\frac{n(n+1)}{2}$ 个损失项，inconsequential 转移只有 n 项，choice points 平均获得 $\frac{n+1}{2}$ 倍重要度[^src-gloeckle-2024-multi-token-prediction]。
2. **互信息放大**（§5.2、Appendix L.2）：2-token 目标的交叉熵满足 $H(p_X, q_X) + H(p_Y, q_Y) = H(p_{X\mid Y}, q_{X\mid Y}) + 2\,I_{p\|q}(X;Y) + H(p_{Y\mid X}, q_{Y\mid X})$，相对互信息 $I_{p\|q}(X;Y)$ 的权重为 2，迫使表示编码约束后续文本的决策而非纯局部风格变体[^src-gloeckle-2024-multi-token-prediction]。
3. **Induction / 算法推理**（§4.1、§4.2）：儿童故事改名任务上，2-token 损失大幅改善 $\leq$30M 非嵌入参数模型的 induction 能力，$\geq$100M 后优势消失；多项式算术任务上 OOD 泛化提升，且把模型从 30M 扩到 100M（约 3 倍）的增益小于把 next-token 换成 multi-token 损失（图 8、S16）[^src-gloeckle-2024-multi-token-prediction]。

该方法与 [[sparse-teacher-forcing|Sparse Teacher Forcing]] / DCRNN scheduled sampling 同属缓解教师强制与自回归错配的方法。Appendix L.1 指出，scheduled sampling 因文本离散难以用于语言建模，交错真值与采样 token 易产生不连贯序列，且为 RNN 设计导致难以并行；multi-token 预测保持并行训练且不改输入[^src-gloeckle-2024-multi-token-prediction]。

## 局限

该方法不保证选择题与似然基准提升，n 需随数据与词表分布调整（图 5、§3.4）[^src-gloeckle-2024-multi-token-prediction]。对已按 next-token 收敛的模型（如 Llama-2）直接多 token 微调收益有限（Table S6；论文归因于「changes the initialization too brutally」）[^src-gloeckle-2024-multi-token-prediction]。增益随数据量增大而收窄：摘要平均 ROUGE-L 增量在 200B→500B 时由 +0.51/+0.46 缩至 +0.28/+0.31（表 S9）；GSM8K 在 200B tokens 时 n=2 更好，500B 后次序翻转（图 S13）[^src-gloeckle-2024-multi-token-prediction]。

## 工程落地：Qwen3-Next 的原生 MTP（2025-09）

阿里 Qwen3-Next 将 MTP 作为原生组件，同时提升预训练性能与投机解码接受率，并针对多步推理采用「训练-推理一致」的多步训练[^src-qwen3-next-blog-2025-09]。部署参数支持 SGLang `NEXTN`（3 步、topk 1、4 个 draft tokens）或 vLLM `qwen3_next_mtp`（2 个 speculative tokens）；HF transformers 未提供 MTP，两源均未给出模块数与内部结构[^src-qwen3-next-modelcard-2025-09]。不同于 Gloeckle et al. 仅作辅助预训练任务，此处采用「训练侧联合目标 + 推理侧 draft」双用途原生设计。架构语境见 [[qwen3-next-hybrid]]。

## 相关

包含 [[source-gloeckle-2024-multi-token-prediction]] 源论文摘要、[[self-speculative-decoding]] 多头加速推理、[[sparse-teacher-forcing]] DSR 中的间歇真值重校准，以及 [[moirai-moe]] 时序基础模型中的 next-token（patch）预训练对照。

## 引用

[^src-gloeckle-2024-multi-token-prediction]: [[source-gloeckle-2024-multi-token-prediction]]
[^src-qwen3-next-blog-2025-09]: [[source-qwen3-next-blog-2025-09]]
[^src-qwen3-next-modelcard-2025-09]: [[source-qwen3-next-modelcard-2025-09]]
