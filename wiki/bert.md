---
title: "BERT：双向 Transformer 编码器预训练"
type: technique
tags:
  - bert
  - pre-training
  - masked-language-model
  - transformer
  - transfer-learning
  - nlp
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# BERT：双向 Transformer 编码器预训练

BERT 是 encoder-only 的两阶段框架：预训练阶段用 masked language model（MLM）与 next sentence prediction（NSP）两个无监督任务训练多层双向 Transformer encoder，微调阶段在预训练参数上接输出层并端到端微调全部参数（§1、§3）[^src-devlin-2019-bert]。同一套预训练参数经不同输入拼接与输出头覆盖单句分类、句对分类、抽取式问答与序列标注，任务间不修改骨干结构（§3.2、Figure 4）[^src-devlin-2019-bert]。与 [[generative-pretraining|GPT-1]] 的差异集中在注意力可见范围与预训练目标，与 ELMo 的差异在 feature-based 与 fine-tuning 两条迁移路线以及双向深度（§1、§5.1、A.4）[^src-devlin-2019-bert]。

## 模型规格

模型分为两种规模：BERT$_{\text{BASE}}$（$L=12$、$H=768$、$A=12$，110M 参数）与 BERT$_{\text{LARGE}}$（$L=24$、$H=1024$、$A=16$，340M 参数），FFN 内层统一为 $4H$（§3）[^src-devlin-2019-bert]。BASE 的规模对齐 OpenAI GPT，使两者对比严格落在注意力掩码与预训练任务上（§3）[^src-devlin-2019-bert]。词表为 30,000 个 WordPiece；每条序列以 `[CLS]` 开头，句对用 `[SEP]` 分隔，每个 token 均叠加可学习的 A/B 段嵌入（§3）[^src-devlin-2019-bert]。"句子"指任意连续文本片段，不限于语言学句子；"序列"指送入模型的单条 token 序列，可含一或两句，上限 512 token（§3）[^src-devlin-2019-bert]。

`[CLS]` 有两种用途：预训练时其末层向量 $C$ 接 NSP 二分类头；微调时同一位置作为分类任务的序列聚合表示并接入分类层（§3、§3.1）[^src-devlin-2019-bert]。未经微调的 $C$ 仅受 NSP 目标训练，不构成有意义的句子表示（§3.1 脚注 6）[^src-devlin-2019-bert]。SQuAD 2.0 中 `[CLS]` 还兼任"无答案"跨度的起点与终点（§4.3）[^src-devlin-2019-bert]。

## 预训练任务

### Masked LM

标准条件语言模型只能从左到右或从右到左训练：若允许双向条件，多层上下文里每个词会间接"看到自己"，目标词可被平凡预测（§3.1）[^src-devlin-2019-bert]。BERT 随机遮住输入中 15% 的 WordPiece，仅对这些位置预测原词而不重建整个输入，不同于去噪自编码器的全量重建目标（§3.1）[^src-devlin-2019-bert]。被选位置的替换策略是 80% 换成 `[MASK]`、10% 换成随机 token、10% 保持原词不变；后两者使编码器无法判断哪些位置会被预测或已被替换，从而对每个输入 token 都维持分布式上下文表示（§3.1、A.1）[^src-devlin-2019-bert]。随机替换只占全部 token 的 1.5%（15% 中的 10%），这一比例不损害语言理解能力（A.1）[^src-devlin-2019-bert]。

$$\begin{aligned}
\mathcal{M} &\subset \{1,\ldots,T\},\quad |\mathcal{M}|=\lfloor 0.15T\rfloor && \text{// } [T]\xrightarrow{\text{均匀抽 }15\%\text{ 位置}}[0.15T]\ \text{——每条序列独立抽，WordPiece 级，不避开子词}\\
\tilde{x}_i &= \begin{cases}\text{[MASK]} & u_i<0.8\\ \text{random token} & 0.8\le u_i<0.9\\ x_i & u_i\ge 0.9\end{cases},\quad i\in\mathcal{M} && \text{// } u_i\sim U(0,1)\ \text{——被选位置三选一；随机替换约占全部 token 的 }1.5\%\\
\mathcal{L}_{\mathrm{MLM}} &= -\frac{1}{|\mathcal{M}|}\sum_{i\in\mathcal{M}}\log P(x_i\mid \tilde{x}) && \text{// } [0.15T,|\mathcal{V}|]\xrightarrow{\text{取目标词项、对位置取平均}}\mathbb{R}\ \text{——监督信号始终是原始 token，不是替换后的 token}
\end{aligned}$$

代价是预训练与微调不匹配：`[MASK]` 在微调阶段从不出现（§3.1）[^src-devlin-2019-bert]。另一个代价是每个 batch 只有 15% 的位置产生监督，导致 MLM 收敛略慢于逐词预测的从左到右模型，但在绝对精度上 MLM 几乎立刻超过后者（C.1、Figure 5）[^src-devlin-2019-bert]。

### Next Sentence Prediction

NSP 从单语语料构造二分类样本：50% 的 B 是 A 的真实下一句（IsNext），50% 是从语料随机抽取的句子（NotNext）；$C$ 接二分类头（§3.1）[^src-devlin-2019-bert]。预训练输入序列由两段连续文本拼成，合计不超过 512 token，第一段用 A 嵌入、第二段用 B 嵌入（A.2）[^src-devlin-2019-bert]。最终模型在 NSP 上达到 97%–98% 准确率（§3.1 脚注 5）[^src-devlin-2019-bert]。预训练总损失是 MLM 平均似然与 NSP 平均似然之和（A.2）[^src-devlin-2019-bert]。

消融表明 NSP 必不可少：去掉 NSP 后 MNLI、QNLI、SQuAD 均下降（Table 5，见下）[^src-devlin-2019-bert]。NSP 让模型理解句间关系，服务 QA 与 NLI 这类句对任务（§3.1）[^src-devlin-2019-bert]。

## 数据流与维度变换

**维度符号**：$T$ 序列 token 数（预训练 90% 步数取 128、其余 10% 取 512，上限 512）；$d$ 隐藏宽度（768/1024）；$n$ encoder 层数（12/24）；$h$ 注意力头数（12/16）；$d_h=d/h$ 单头通道维；$d_{ff}=4d$ FFN 内层宽度；$|\mathcal{V}|=30{,}000$ WordPiece 词表；$s\in\{A,B\}$ 段标识（段嵌入表 2 行）；$p$ 位置下标（位置表最大 512 行）；$K$ 分类标签数；$T_p$ 问答段落 token 数。形状记作 $[\text{行},\text{列}]$。

### Stage 1 · 拼序列并合成输入嵌入

单句任务取 `[CLS] tok [SEP]`；句对任务用 `[SEP]` 分隔 A、B 两段，并以 `[SEP]` 收尾（§3、Figure 2）[^src-devlin-2019-bert]。三张查找表分别对应 token 身份、句子归属与位置，逐位置相加得到输入表示（§3、Figure 2）[^src-devlin-2019-bert]。

$$\begin{aligned}
x &= [\text{[CLS]};\ \mathrm{tok}(A);\ \text{[SEP]};\ \mathrm{tok}(B);\ \text{[SEP]}] && \text{// } [T]\ \text{——单句时退化为 [CLS] tok [SEP]；[SEP] 兼作段界与序列终结符}\\
E_{tok} &= W_{tok}[x] && \text{// } [T]\xrightarrow{\text{按词表 id 取行}}[T,d]\ \text{——每个 WordPiece 一个向量}\\
E_{seg} &= W_{seg}[s] && \text{// } [T]\xrightarrow{\text{按 } s\in\{A,B\} \text{ 取行}}[T,d]\ \text{——同一段内所有位置共享一行}\\
E_{pos} &= W_{pos}[p] && \text{// } [T]\xrightarrow{\text{按位置 } 0..T{-}1 \text{ 取行}}[T,d]\ \text{——位置表最大 512 行，每个位置一个向量}\\
E &= E_{tok}+E_{seg}+E_{pos} && \text{// } [T,d]+[T,d]+[T,d]\to[T,d]\ \text{——三路逐元素相加，同一位置叠加三类信息}\\
h^{0} &= \operatorname{Dropout}\!\left(\operatorname{LN}(E)\right) && \text{// } [T,d]\to[T,d]\ \text{——按官方实现先 LN 再 dropout（p=0.1）；论文 §3 只写三项相加，LN 由原 Transformer block 带入 [INFERENCE]}
\end{aligned}$$

### Stage 2 · 逐层双向编码

每层为标准 Transformer encoder block（见 [[transformer-architecture]]）：包含多头自注意力与逐位置 FFN，两子层各带残差与 LayerNorm（§3 指向 Vaswani et al., 2017；post-LN 排布为原实现，标注为推断）[^src-devlin-2019-bert][INFERENCE]。自注意力不加 causal mask，除 padding 外所有位置互相可见，每个 token 在每一层都同时读到左右上下文（§3、Figure 3）[^src-devlin-2019-bert]。

$$\begin{aligned}
Q &= h^{l-1}W_Q,\quad K=h^{l-1}W_K,\quad V=h^{l-1}W_V && \text{// } [T,d]\times[d,d]\to[T,d]\ \text{——每层三组投影，多头切分后单头为 }[T,d_h]\\
A &= \operatorname{softmax}\!\left(QK^{\top}/\sqrt{d_h}\right) && \text{// } [T,\bcancel{d_h}]\times[\bcancel{d_h},T]\to[T,T]\xrightarrow{\text{沿 }T\text{ 维 softmax}}[T,T]\ \text{——第 }i\text{ 行是 token }i\text{ 对全部 }T\text{ 个位置的读权，无 causal mask}\\
O &= AV && \text{// } [T,\bcancel{T}]\times[\bcancel{T},d_h]\to[T,d_h]\ \text{——按权重汇总全部位置的值，含右侧上下文}\\
\hat{h}^{l} &= \operatorname{LN}\!\left(h^{l-1}+\operatorname{Concat}(O_1,\ldots,O_h)W_O\right) && \text{// } h\times[T,d_h]\xrightarrow{\text{拼接}}[T,d]\ \text{——多头结果拼回 }d\text{ 后投影，残差相加}\\
h^{l} &= \operatorname{LN}\!\left(\hat{h}^{l}+\operatorname{GELU}(\hat{h}^{l}W_1)W_2\right) && \text{// } [T,\bcancel{d}]\times[\bcancel{d},4d]\to[T,4d]\times[\bcancel{4d},d]\to[T,d]\ \text{——逐位置两层 FFN，位置间不交互}\\
h^{n} &= \operatorname{Block}^{n}(h^{0}) && \text{// } [T,d]\to[T,d]\ \text{——重复 }n\text{ 次，输出仍是每位置一个 }d\text{ 维向量}
\end{aligned}$$

### Stage 3 · 预训练双头与损失

$$\begin{aligned}
H &= h^{n} && \text{// } [T,d]\ \text{——末层全部位置的隐藏状态}\\
P^{\mathrm{MLM}} &= \operatorname{softmax}(HW_{vocab}^{\top}) && \text{// } [T,\bcancel{d}]\times[\bcancel{d},|\mathcal{V}|]\to[T,|\mathcal{V}|]\ \text{——逐位置给全词表分布，只有 }\mathcal{M}\text{ 行进入损失}\\
\mathcal{L}_{\mathrm{MLM}} &= -\frac{1}{|\mathcal{M}|}\sum_{i\in\mathcal{M}}\log P^{\mathrm{MLM}}_{i,x_i} && \text{// } [T,|\mathcal{V}|]\xrightarrow{\text{取 }0.15T\text{ 行}}[0.15T,|\mathcal{V}|]\to\mathbb{R}\ \text{——梯度只从被遮位置回传}\\
P^{\mathrm{NSP}} &= \operatorname{softmax}(CW_{NSP}^{\top}) && \text{// } [d]\times[d,2]\to[2]\ \text{——}C\text{ 是 }H\text{ 的 }[CLS]\text{ 行，二分类 IsNext/NotNext}\\
\mathcal{L} &= \mathcal{L}_{\mathrm{MLM}}+\mathcal{L}_{\mathrm{NSP}} && \text{// } \mathbb{R}+\mathbb{R}\to\mathbb{R}\ \text{——两项平均似然之和（A.2）}
\end{aligned}$$

### Stage 4 · 微调按任务换头

骨干前向与 Stage 1–2 相同，只替换输出层；所有预训练参数参与微调，每个下游任务各自保存一份微调模型（§3.2）[^src-devlin-2019-bert]。

**支路 A：单句/句对分类**。输入为 `[CLS] tok [SEP]` 或 `[CLS] tok(A) [SEP] tok(B) [SEP]`，取 $C$ 接线性层；新增参数只有 $W\in\mathbb{R}^{K\times H}$（§4.1）[^src-devlin-2019-bert]。

$$\begin{aligned}
P(y\mid x) &= \operatorname{softmax}(CW^{\top}) && \text{// } [d]\times[d,K]\to[K]\ \text{——一句或一对句子压缩成一个标签分布}\\
\mathcal{L}_{\mathrm{cls}} &= -\log P(y\mid x) && \text{// } [K]\to\mathbb{R}\ \text{——标准分类交叉熵}
\end{aligned}$$

**支路 B：抽取式问答（SQuAD）**。问题用 A 段、段落用 B 段，新增起点向量 $S\in\mathbb{R}^{H}$ 与终点向量 $E\in\mathbb{R}^{H}$（§4.2）[^src-devlin-2019-bert]。

$$\begin{aligned}
p_i^{\mathrm{start}} &= \frac{\exp(S\cdot T_i)}{\sum_j \exp(S\cdot T_j)} && \text{// } [d]\cdot[T_p,d]\to[T_p]\xrightarrow{\text{沿段落位置 softmax}}[T_p]\ \text{——段内每个 token 成为答案起点的概率}\\
p_j^{\mathrm{end}} &= \frac{\exp(E\cdot T_j)}{\sum_k \exp(E\cdot T_k)} && \text{// } [d]\cdot[T_p,d]\to[T_p]\ \text{——同一形式，终点参数独立}\\
(i^{\star},j^{\star}) &= \arg\max_{j\ge i}\left(S\cdot T_i+E\cdot T_j\right) && \text{// } [T_p,T_p]\xrightarrow{\text{上三角内取最大}}[1]\ \text{——候选跨度打分，约束终点不早于起点}\\
\mathcal{L}_{\mathrm{qa}} &= -\log p^{\mathrm{start}}_{i^{\star}}-\log p^{\mathrm{end}}_{j^{\star}} && \text{// } \mathbb{R}+\mathbb{R}\to\mathbb{R}\ \text{——正确起止位置的对数似然之和}
\end{aligned}$$

SQuAD 2.0 把无答案问题视为起止都在 `[CLS]` 的跨度，打分 $s_{\mathrm{null}}=S\cdot C+E\cdot C$，与最佳非空跨度 $\hat{s}_{i,j}$ 比较，仅当 $\hat{s}_{i,j}>s_{\mathrm{null}}+\tau$ 时预测非空；阈值 $\tau$ 在 dev 上按 F1 选取（§4.3）[^src-devlin-2019-bert]。

**支路 C：多选（SWAG）**。给定句子与 4 个候选续写，各拼成一条 (A, B) 序列独立前向，用 $C$ 与一个向量的点积为各候选打分，再在 4 个候选间取 softmax（§4.4）[^src-devlin-2019-bert]。

$$\begin{aligned}
s_k &= C_k\cdot w && \text{// } [4,d]\times[d,1]\to[4]\ \text{——4 条序列各取自己的 }[CLS]\text{ 表示}\\
P(k) &= \operatorname{softmax}(s)_k && \text{// } [4]\xrightarrow{\text{沿候选维 softmax}}[4]\ \text{——候选间竞争，不是逐 token 分类}
\end{aligned}$$

### 结构要点

句对不另建 cross-attention，把两段拼成一条序列后做自注意力，等价于两句之间双向 cross attention，编码与交互合成一步（§3.2）[^src-devlin-2019-bert]。任务差异只落在输入拼法与输出头：分类读 $C$，序列标注与问答读各 token 的 $T_i$，多选复制 4 条序列；骨干与超参基本不动（§3.2、Figure 4）[^src-devlin-2019-bert]。`[CLS]` 的语义依赖训练目标：预训练时服务 NSP，微调时才压成任务相关的序列摘要，未微调的 $C$ 不是句子表示（§3.1 脚注 6）[^src-devlin-2019-bert]。

## 与 GPT-1、ELMo 的对照

与 GPT-1 相比，两者均采用"预训练 + 全参数微调"，BASE 规模对齐（110M），差异在注意力可见范围与预训练目标。GPT-1 是 decoder-only，每个 token 只能看左侧，预训练用标准从左到右 LM、每个位置都有监督；BERT 是 encoder-only，每层双向，预训练用 MLM、每 batch 只监督 15%（§1、§3、Figure 3；GPT-1 机制与数字见 [[generative-pretraining]] 与 [[source-radford-2018-gpt]]）[^src-devlin-2019-bert]。A.4 另列四项训练差异：GPT 只用 BooksCorpus（800M 词），BERT 加英文 Wikipedia（合计 3.3B 词）；GPT 的 `[SEP]`/`[CLS]` 只在微调时引入，BERT 在预训练中学习它们与 A/B 段嵌入；GPT 每 batch 32,000 词、BERT 128,000 词；GPT 所有微调任务共用 5e-5 学习率，BERT 按任务在 dev 上选（A.4）[^src-devlin-2019-bert]。Table 5 的 "LTR & No NSP" 行用 BERT 的数据、输入表示与微调方案复现从左到右模型，且微调阶段同样保留左向约束（§5.1），五项任务全面低于 MLM 版本，差距最大的是 MRPC（86.7→77.5）与 SQuAD（88.5→77.8）（§5.1、Table 5）[^src-devlin-2019-bert]。

ELMo 采用 feature-based 路线，独立训练从左到右与从右到左两个 LSTM，把每个 token 的两侧表示拼接后作为固定特征交给任务模型；BERT 采用 fine-tuning，双向性发生在每一层内部（§1、§2.1、Figure 3）[^src-devlin-2019-bert]。分别训练 LTR/RTL 再拼接有三条缺陷：成本是单个双向模型的两倍；对 QA 不直观，因为 RTL 模型无法让答案条件于问题；浅层拼接严格弱于每层都能结合双向上下文的深双向模型（§5.1）[^src-devlin-2019-bert]。BERT 同时保留 feature-based 用法：固定表示接两层 BiLSTM 做 NER，最佳组合（末四层拼接）取得 96.1 dev F1，只比全参数微调的 96.4 低 0.3（§5.3、Table 7）[^src-devlin-2019-bert]。

## 预训练设置

语料为 BooksCorpus（800M 词）与英文 Wikipedia（2,500M 词，只取正文段落，丢弃列表、表格与标题），合计约 3.3B 词；选用文档级语料而非打乱的句级语料（如 Billion Word Benchmark），以抽取长连续序列（§3.1）[^src-devlin-2019-bert]。每 batch 256 条序列、每条 512 token，即 128,000 token/batch，训练 1,000,000 步，约合该语料 40 个 epoch（A.2）[^src-devlin-2019-bert]。优化器为 Adam，学习率 1e-4、$\beta_1=0.9$、$\beta_2=0.999$，L2 权重衰减 0.01，前 10,000 步 warmup 后线性衰减；所有层 dropout 0.1，激活函数用 GELU（A.2）[^src-devlin-2019-bert]。

为控制注意力随序列长度的二次开销，前 90% 步数以长度 128 预训练，最后 10% 步数切换至 512 学习位置嵌入（A.2）[^src-devlin-2019-bert]。这与同节“256 条 × 512 token = 128,000 token/batch、1M 步、约 40 epoch”的口径并列：若 90% 步数按 128 token 计，每 batch 仅 32,768 token，40 epoch 的换算只在全程 512 口径下成立；文中未说明两者如何统一（A.2）[^src-devlin-2019-bert]。BASE 在 4 块 Cloud TPU（Pod 配置，共 16 个 TPU 芯片）上训练，LARGE 在 16 块 Cloud TPU（共 64 个芯片）上训练，各耗时 4 天（A.2）[^src-devlin-2019-bert]。

微调仅调整 batch size、学习率与 epoch 数，dropout 保持 0.1；搜索范围为 batch 16/32、学习率 5e-5/3e-5/2e-5、epoch 2/3/4（A.3）[^src-devlin-2019-bert]。数据量 10 万以上的任务对超参不敏感，小数据任务敏感（A.3）[^src-devlin-2019-bert]。

## 实验

评测覆盖 11 项任务：GLUE 的 8 项（MNLI、QQP、QNLI、SST-2、CoLA、STS-B、MRPC、RTE；WNLI 被排除）、SQuAD 1.1、SQuAD 2.0 与 SWAG（§4、Table 1）[^src-devlin-2019-bert]。

**GLUE（Table 1，test，单模型单任务）**：平均分 BERT$_{\text{BASE}}$ 79.6、BERT$_{\text{LARGE}}$ 82.1，OpenAI GPT 75.1，Pre-OpenAI SOTA 74.0，BiLSTM+ELMo+Attn 71.0（§4.1）[^src-devlin-2019-bert]。论文正文称 BASE/LARGE 相对此前 SOTA 平均提升 4.5 与 7.0 点；按 Table 1 的 Pre-OpenAI SOTA 行 74.0 计算应为 5.6 与 8.1，4.5/7.0 实为相对 OpenAI GPT（75.1）的差值，两处口径并列（§4.1、Table 1）[^src-devlin-2019-bert]。GLUE 微调统一 batch 32、3 epoch，学习率在 5e-5/4e-5/3e-5/2e-5 中按 dev 选（§4.1）[^src-devlin-2019-bert]。逐项（BASE/LARGE）：MNLI-m 84.6/86.7、MNLI-mm 83.4/85.9、QQP 71.2/72.1、QNLI 90.5/92.7、SST-2 93.5/94.9、CoLA 52.1/60.5、STS-B 85.8/86.5、MRPC 88.9/89.3、RTE 66.4/70.1（Table 1）[^src-devlin-2019-bert]。官方榜上 BERT$_{\text{LARGE}}$ 得 80.5，OpenAI GPT 为 72.8，摘要报 +7.7 点；Table 1 的平均分与官方 GLUE 分不同，因为 Table 1 排除了 WNLI，且 BERT 提交时对 WNLI 一律预测多数类（Table 1 题注、§4.1、B.1）[^src-devlin-2019-bert]。

**SQuAD 1.1（Table 2）**：BERT$_{\text{BASE}}$ 单模型 dev 80.8/88.5（EM/F1），BERT$_{\text{LARGE}}$ 单模型 dev 84.1/90.9，7 模型集成 dev 85.8/91.8；经 TriviaQA 微调后，单模型 test 85.1/91.8，集成 test 87.4/93.2（Table 2）[^src-devlin-2019-bert]。同期榜首集成 test 86.0/91.7，人类 test 82.3/91.2；集成较榜首高 1.5 F1，摘要报告 93.2（+1.5）（Table 2、摘要）[^src-devlin-2019-bert]。集成由 7 个不同预训练 checkpoint 与微调种子的系统组成（Table 2 题注）[^src-devlin-2019-bert]。

**SQuAD 2.0（Table 3）**：BERT$_{\text{LARGE}}$ 单模型 dev 78.7/81.9、test 80.0/83.1，此前最好单模型 test 74.8/78.0，提升 5.1 F1；人类 test 86.9/89.5（Table 3、§4.3）[^src-devlin-2019-bert]。训练 2 epoch、学习率 5e-5、batch 48，未用 TriviaQA（§4.3）[^src-devlin-2019-bert]。

**SWAG（Table 4）**：BERT$_{\text{BASE}}$ dev 81.6，BERT$_{\text{LARGE}}$ dev 86.6、test 86.3；ESIM+GloVe 51.9/52.7，ESIM+ELMo 59.1/59.2，OpenAI GPT test 78.0；人类专家 85.0、五人标注 88.0（Table 4）[^src-devlin-2019-bert]。LARGE 比 ESIM+ELMo 高 27.1 点、比 GPT 高 8.3 点（§4.4）[^src-devlin-2019-bert]。微调 3 epoch、学习率 2e-5、batch 16（§4.4）[^src-devlin-2019-bert]。

**NER（Table 7，CoNLL-2003，dev/test F1）**：全参数微调 BERT$_{\text{LARGE}}$ 96.6/92.8、BERT$_{\text{BASE}}$ 96.4/92.4；ELMo 95.7/92.2，CVT test 92.6，CSE test 93.1（Table 7）[^src-devlin-2019-bert]。固定特征（BASE）逐层：仅嵌入 91.0、倒数第二层 95.6、末层 94.9、末四层加权和 95.9、末四层拼接 96.1、全部 12 层加权和 95.5（Table 7）[^src-devlin-2019-bert]。

## 消融

**预训练任务（Table 5，BASE，dev）**：完整模型 MNLI-m/QNLI/MRPC/SST-2/SQuAD F1 为 84.4/88.4/86.7/92.7/88.5；去掉 NSP 后为 83.9/84.9/86.5/92.6/87.9，QNLI 降 3.5 点；改为从左到右且无 NSP 为 82.1/84.3/77.5/92.1/77.8，MRPC 与 SQuAD 跌幅最大；在该 LTR 模型上加随机初始化的 BiLSTM 后 SQuAD 回升至 84.9，仍低于 88.5，且 GLUE 四项中三项下降（Table 5、§5.1）[^src-devlin-2019-bert]。从左到右模型在 token 级任务上缺少右侧上下文，加 BiLSTM 只能部分补偿（§5.1）[^src-devlin-2019-bert]。

**模型规模（Table 6，dev，5 次随机重启平均）**：$L/H/A$ 从 3/768/12 增至 24/1024/16，MNLI-m 77.9→86.6、MRPC 79.8→87.8、SST-2 88.4→93.7，held-out 训练数据的 MLM 困惑度 5.84→3.23；中间档 6/768/3、6/768/12、12/768/12、12/1024/16 逐档单调改善（Table 6、§5.2）[^src-devlin-2019-bert]。充分预训练后，模型放大对极小规模下游任务同样带来一致提升（§5.2）[^src-devlin-2019-bert]。

**掩码策略（Table 8，BASE，dev）**：80/10/10 的 MNLI/NER 微调/NER 特征为 84.2/95.4/94.9；100/0/0 为 84.3/94.9/94.0；80/0/20 为 84.1/95.2/94.6；80/20/0 为 84.4/95.2/94.7；0/20/80 为 83.7/94.8/94.6；0/0/100 为 83.6/94.9/94.6（Table 8）[^src-devlin-2019-bert]。微调对掩码策略不敏感；只用 `[MASK]` 时特征式 NER 降幅最大（94.0 对 94.9），只用随机替换也明显更差（§C.2）[^src-devlin-2019-bert]。

**训练步数（C.1、Figure 5）**：1M 步比 500k 步在 MNLI 上约多 1.0 点准确率；MLM 收敛略慢于 LTR，但绝对精度几乎立即反超（C.1）[^src-devlin-2019-bert]。

## 边界

`[MASK]` 造成的预训练-微调不匹配仅被 80/10/10 缓解而未消除；特征式用法下该不匹配被放大，NER 的 100% `[MASK]` 配置降至 94.0（§3.1、Table 8）[^src-devlin-2019-bert]。因每 batch 仅预测 15% 的 token，MLM 比逐词 LM 收敛更慢，需要更长预训练，论文采用 1M 步（§3.1、A.1、A.2、C.1）[^src-devlin-2019-bert]。GLUE 的 Table 1 平均分（82.1）与官方榜分（80.5）口径不同：前者排除 WNLI，后者含 WNLI 且对 WNLI 预测多数类，两数并列而不互相替代（Table 1 题注、§4.1、B.1）[^src-devlin-2019-bert]。论文仅报告单任务微调；脚注提及多任务微调（如先用 MNLI 再 RTE）可能继续提升，但未做系统实验（B.1 脚注 14）[^src-devlin-2019-bert]。预训练语料仅有英文 BooksCorpus 与英文 Wikipedia，未给出跨语言或多语言评测（§3.1、A.2）[^src-devlin-2019-bert]。BERT$_{\text{LARGE}}$ 在小数据 GLUE 任务上微调不稳定，论文对同一预训练 checkpoint 多次随机重启并在 dev 上选出最优模型提交评测服务器，Table 1 为该次提交的 test 结果（§4.1、脚注 9）[^src-devlin-2019-bert]。SQuAD 1.1 的最好成绩结合了 7 模型集成与 TriviaQA 外部数据；不用外部数据的单模型 dev F1 为 90.9（Table 2）[^src-devlin-2019-bert]。

## 脚注

[^src-devlin-2019-bert]: [[source-devlin-2019-bert]]
