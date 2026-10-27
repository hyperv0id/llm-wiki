---
title: "生成式预训练（GPT-1 两阶段范式）"
type: technique
tags:
  - gpt
  - pre-training
  - language-model
  - transfer-learning
  - transformer
created: 2026-10-27
last_updated: 2026-10-27
source_count: 2
confidence: medium
status: active
---

# 生成式预训练（GPT-1 两阶段范式）

生成式预训练是先在无标注语料上以语言建模目标训练高容量 Transformer decoder，再在有标注目标任务上判别式微调的半监督框架（§1–§2）[^src-radford-2018-gpt]。该框架回答了两个未决问题：什么预训练目标能学到可迁移表示（此前 LM、机器翻译、篇章连贯各有胜负），以及如何把表示迁移到目标任务（此前多改用任务专属架构、复杂学习方案或加辅助目标）（§1）[^src-radford-2018-gpt]。GPT-1 给出的方案是：预训练采用标准 LM 目标，迁移依靠任务特定的**输入变换**——把结构化输入折成单一 token 序列，模型架构在 12 个任务间几乎无需改动（摘要、§3.3）[^src-radford-2018-gpt]。同一两阶段框架下，把预训练目标换成双向掩码语言建模、骨干换成 encoder-only 的路线见 [[bert]]。

## 两阶段训练目标

**Stage 1（无监督预训练）** 最大化似然 $\mathcal{L}_1(\mathcal{U}) = \sum_i \log P(u_i \mid u_{i-k},\ldots,u_{i-1};\Theta)$，$k$ 为上下文窗口，$P$ 由多层 Transformer decoder 参数化（Eq. 1）[^src-radford-2018-gpt]。

**Stage 2（有监督微调）** 由预训练参数初始化，末层激活接新增线性层 $W_y$ 预测标签并最大化 $\mathcal{L}_2(C) = \sum_{(x,y)} \log P(y \mid x^1,\ldots,x^m)$（Eq. 3–4）；实际优化 $\mathcal{L}_3(C) = \mathcal{L}_2(C) + \lambda \cdot \mathcal{L}_1(C)$，$\lambda = 0.5$（Eq. 5，§4.1）[^src-radford-2018-gpt]。辅助 LM 目标改善了监督模型泛化并加速收敛（§3.2）[^src-radford-2018-gpt]。

## 数据流与维度变换

**维度符号**：$N$ 输入 token 数（预训练 512）；$d$ 残差流宽度（768）；$n$ block 层数（12）；$h$ 注意力头数（12）；$d_{ff}$ 前馈内层宽度（3072）；$|\mathcal{V}|$ BPE 词表（40,000 merges）；$m$ 输入序列长度（微调记号）；$|\mathcal{Y}|$ 任务标签数。形状记 $[\text{行}, \text{列}]$。

### Stage 1 · 语言建模预训练

$$\begin{aligned}
h^0 &= U W_e + W_p && \text{// } [N, \bcancel{|\mathcal{V}|}]\times[\bcancel{|\mathcal{V}|}, d] + [N, d] \to [N, d]\ \text{——token 嵌入与位置嵌入逐位置相加，得残差流初值（Eq. 2）}\\
h^l &= \operatorname{TransformerBlock}(h^{l-1}) && \text{// } [N, d] \to [N, d]\ \text{——12 层 decoder block：掩码多头自注意力（每位置只看 } \le i \text{）+ 逐位置 FFN，post-LN 残差}\\
P(u) &= \operatorname{softmax}(h^n W_e^{\top}) && \text{// } [N, \bcancel{d}]\times[\bcancel{d}, |\mathcal{V}|] \to [N, |\mathcal{V}|]\ \text{——输出投影复用嵌入矩阵 } W_e \text{（weight tying），逐 token 给全词表分布（Eq. 2）}\\
\mathcal{L}_1 &= \sum_i \log P(u_i \mid u_{i-k}, \ldots, u_{i-1}) && \text{// } [N, |\mathcal{V}|] \xrightarrow{\text{取目标 token 项、沿 } N \text{ 维求和}} \mathbb{R}\ \text{——标准 LM 目标，Adam 优化（Eq. 1）}
\end{aligned}$$

### Stage 2 · 判别式微调

微调输入先按 Stage 3 的变换折成序列，再走与预训练相同的 12 层前向：

$$\begin{aligned}
h_l^m &\in \mathbb{R}^d && \text{// } h^0 \in [m, d] \xrightarrow{\text{12 层 block，取末 token 行}} [d]\ \text{——最后一个 token 在最终 block 的激活，作序列表示（§3.2）}\\
P(y \mid x^1, \ldots, x^m) &= \operatorname{softmax}(h_l^m W_y) && \text{// } [\bcancel{d}]\times[d, |\mathcal{Y}|] \to [|\mathcal{Y}|]\ \text{——仅新增 } W_y \text{ 与分隔符嵌入（Eq. 3）}\\
\mathcal{L}_3 &= \mathcal{L}_2 + \lambda \mathcal{L}_1 && \text{// } \lambda = 0.5\ \text{——判别似然与辅助 LM 目标联合优化（Eq. 5）}
\end{aligned}$$

### Stage 3 · 任务特定输入变换

预训练模型只处理连续 token 序列，而 NLI 为句对、QA 为文档-问题-答案三元组。GPT-1 不为各任务添加专用模块，而是按 traversal 风格把输入拼成单序列，以 `$` 为分隔符，首尾加入随机初始化的 $\langle s\rangle$、$\langle e\rangle$（§3.3，Figure 1 右）[^src-radford-2018-gpt]：

$$\begin{aligned}
T_{\text{cls}}(x) &= [\langle s\rangle; x; \langle e\rangle] && \text{// 单句分类直接拼接，取末 token } h_l^m \in [d]\ \text{进 } W_y\\
T_{\text{ent}}(p, h) &= [\langle s\rangle; p; \$; h; \langle e\rangle] && \text{// 文本蕴含的 premise-hypothesis 有序对，单次前向取 } h_l^m\\
\bar{h}_l^m &= h_l^m(T(s_1, s_2)) + h_l^m(T(s_2, s_1)) && \text{// 相似度任务无内在顺序：两个顺序各前向一次，}[d] + [d] \to [d]\ \text{逐元素相加后再进 } W_y\\
\hat{y} &= \operatorname{softmax}_k\bigl(h_l^m([\langle s\rangle; z; q; \$; a_k; \langle e\rangle])\bigr) && \text{// QA/常识推理：文档 } z \text{、问题 } q \text{ 与每个答案 } a_k \text{ 独立拼接前向，沿 } K \text{ 个答案维 softmax 归一化}
\end{aligned}$$

输入文本排布承担了任务差异而无需引入新模块，单一任务无关模型由此覆盖 12 个任务（摘要）[^src-radford-2018-gpt]。Zero-shot 实验给 SST-2 例句追加 `very`、比较 `positive`/`negative` 的概率（§5），已是手工构造提示词的雏形；论文当时未提出系统化的 prompt 概念。

## 模型规格与相对原版 Transformer 的差异

模型大体沿用原版 Transformer（见 [[transformer-architecture]]），采用 12 层 decoder-only：掩码自注意力 + 逐位置 FFN，$d = 768$、$h = 12$、$d_{ff} = 3072$（§4.1）[^src-radford-2018-gpt]。相比 [[transformer-architecture]] 的原始配置，改动包括：位置编码从正弦函数改为可学习嵌入；激活函数从 ReLU 换成 GELU；输入采用 40,000 merges 的 BPE；子层排布沿用 post-LN 残差（pre-LN 为 GPT-2 改动，见 [[pre-ln-transformer]]）[^src-radford-2018-gpt][^src-gpt2-2019-language-models]。预训练在 BooksCorpus（7000+ 本未出版书籍，含长连续文本）上运行 100 epochs，batch 64 条 512 token 序列，Adam 峰值学习率 2.5e-4（2000 步线性 warmup + cosine 退火），密集使用 LayerNorm 使 $N(0, 0.02)$ 初始化即可，dropout 0.1，非 bias/gain 权重加 $w = 0.01$ 的修正 L2，语言模型在该语料上 token 级困惑度 18.4（§4.1）[^src-radford-2018-gpt]。微调复用预训练超参，学习率 6.25e-5、batch 32、3 epochs、warmup 占 0.2%（§4.1）[^src-radford-2018-gpt]。

## 实验证据

12 个任务包括 NLI 五个、QA 两个、语义相似度三个、分类两个（Table 1）[^src-radford-2018-gpt]。单模型在 9 个任务上超过此前最好成绩，并在多处超过集成模型（§4.2）[^src-radford-2018-gpt]。

**NLI（Table 2，accuracy，除注明外均为单模型）**：MNLI-m 82.1 / MNLI-mm 81.4（此前最佳 SAN 3x 80.6 / 80.1，+1.5）、SNLI 89.9（此前 ESIM+ELMo 5x 与 CAFE 5x 各 89.3，+0.6）、SciTail 88.3（CAFE 83.3，+5.0）、QNLI 88.1（GenSen 82.3，+5.8）；RTE 56.0，低于 multi-task biLSTM+Attn 的 61.7（§4.2）[^src-radford-2018-gpt]。

**QA 与常识推理（Table 3）**：Story Cloze 86.5（此前 Hidden Coherence Model 77.6，+8.9）；RACE 总 59.0、RACE-m 62.9、RACE-h 57.4（此前 BiAttention MRU 9x 53.3，单模型 +5.7）（§4.2）[^src-radford-2018-gpt]。

**语义相似度与分类（Table 4，GLUE 评测）**：STS-B pc 82.0（ECNU 集成 81.0，+1.0）、QQP F1 70.3（Single-task BiLSTM+ELMo+Attn 66.1，+4.2）、CoLA mc 45.4（此前最佳 35.0，+10.4）；SST-2 acc 91.3 低于 Sparse byte mLSTM 的 93.2，MRPC F1 82.3 低于 TF-KLD 的 86.0，均未达 SOTA；GLUE 总分 72.8（此前最佳 68.9）（§4.2）[^src-radford-2018-gpt]。引言称 GLUE 提升 5.5%，按 Table 4 两数相减为 3.9 点，两处口径见 [[ingest-reports/2026-10-27-gpt|WHY 报告]]数值备注[^src-radford-2018-gpt]。

**消融（Table 5，8 数据集未加权平均）**[^src-radford-2018-gpt]：

| 变体 | Avg. | 相对完整模型的差异（论文报告） |
|---|---|---|
| Transformer w/ aux LM（完整） | 74.7 | — |
| w/o aux LM | 75.0 | 平均略升但分化：MNLI −0.7、QNLI −1.2、RTE −1.6、QQP −0.5，小数据集反而上升（CoLA +2.5、MRPC +2.6、STS-B +1.2、SST-2 +0.7）；趋势为大集合受益于辅助目标、小集合不受益 |
| LSTM w/ aux LM（单层 2048 单元） | 69.1 | −5.6，仅 MRPC 一项超过 Transformer（83.2 对 82.3） |
| Transformer w/o pre-training | 59.9 | −14.8，所有任务受损 |

**层数迁移（Figure 2 左）**：在 MultiNLI 与 RACE 上逐步增加迁移层数：仅迁移嵌入即有提升，每增一层均持续受益，MultiNLI 全量迁移比只迁移嵌入最多高 9%（§5）[^src-radford-2018-gpt]。

**Zero-shot 行为（Figure 2 右，§5）**：不经微调，直接以启发式调用生成模型：CoLA 按平均 token log-prob 阈值判定可接受性，SST-2 追加 `very` 后比较 `positive`/`negative` 概率，RACE 选条件平均 log-prob 最高的答案，DPRD（Winograd）替换代词后比较后续序列 log-prob；四项启发式的归一化性能随预训练更新稳定上升，且 Transformer 的零样本方差低于 LSTM（§5）[^src-radford-2018-gpt]。

## 边界

RTE（2490 样本）上 56.0 低于多任务 biLSTM 的 61.7，推测多任务训练有效但未探索（§4.2）[^src-radford-2018-gpt]；SST-2 与 MRPC 未超各自 SOTA（Table 4）[^src-radford-2018-gpt]；辅助 LM 目标在大数据集有益、小数据集无益甚至略负（Table 5）[^src-radford-2018-gpt]。数据选用 BooksCorpus，以长连续文本学习长程条件依赖；对比 ELMo 使用的 1B Word Benchmark，后者规模相近但按句打乱、破坏长程结构（§4.1）[^src-radford-2018-gpt]。评测仅覆盖四类判别式理解任务，生成与抽取式问答等未纳入范围（§4.2）[^src-radford-2018-gpt]。

## 脚注

[^src-radford-2018-gpt]: [[source-radford-2018-gpt]]
[^src-gpt2-2019-language-models]: [[source-gpt2-2019-language-models]]
