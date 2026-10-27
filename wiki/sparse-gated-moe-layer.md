---
title: "Sparsely-Gated Mixture-of-Experts Layer"
type: technique
tags:
  - mixture-of-experts
  - conditional-computation
  - sparse-activation
  - load-balancing
  - distributed-training
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Sparsely-Gated Mixture-of-Experts Layer

Shazeer et al.（Google Brain，arXiv:1701.06538，ICLR 2017）提出稀疏门控 MoE 层，包含 $n$ 个前馈 expert 与一个可训练门控网络，每个样本只激活其中 $k$ 个。论文首次兑现条件计算（conditional computation）收益，容量提升超 1000 倍，最大 137B 参数、每样本只算 4 个 expert[^src-shazeer-2017-sparse-moe]。

## 问题：条件计算此前为何没做成

稠密模型中容量与计算近似同步增长，训练成本随二者膨胀；条件计算按样本开关大部分网络，理论上使容量与计算解耦[^src-shazeer-2017-sparse-moe]。论文列出此前的四条障碍（§1.1）：GPU 擅长算术不擅长分支，门控必须一次开关大块网络；条件激活稀释 batch，参数搬运摊不平；集群算力与互联带宽之比可达数千比一，embedding 式条件计算被带宽卡死；稀疏化与噪声需要额外损失项。此外，此前最大数据集仅约 60 万张图，撑不起十亿级参数；论文转向数据量充足的文本领域，用于语言建模与翻译[^src-shazeer-2017-sparse-moe]。

## 层定义与 noisy top-k 门控

MoE 层由 $n$ 个 expert 网络 $E_1,\dots,E_n$（采用参数独立的同构单隐层 FFN）与一个输出稀疏 $n$ 维向量的门控网络 $G$ 组成，输出为 $y=\sum_{i=1}^{n}G(x)_i E_i(x)$（式 1），$G(x)_i=0$ 的 expert 无需计算[^src-shazeer-2017-sparse-moe]。软门控版本即 $G_\sigma(x)=\operatorname{Softmax}(xW_g)$（式 2，Jacobs-Jordan 一系）；论文在此基础上增加稀疏与噪声两项，构成 noisy top-k gating（式 3–5），完整计算式见下方 Stage 1–3[^src-shazeer-2017-sparse-moe]。门控随全模型以普通反向传播训练：$k>1$ 时被选中 expert 的门控值对门控权重有非零导数，梯度回传至门控输入，不同于 Bengio et al. 2015 的布尔门加 REINFORCE 路线（§2.1）[^src-shazeer-2017-sparse-moe]。

## 数据流与维度变换

> **维度符号**：$x\in\mathbb{R}^d$ 是单个时间步的输入（LM 中 $d=512$，即上一层 LSTM 在该位置的隐状态）；$n$ 为 expert 数（4–131072）；每个 expert 是同构单隐层 FFN，隐层宽 $h$ 记 expert 容量（1B word 低算力系列 $h=1024$，每 expert 约 1M 参数；MT 系列每 expert 约 2M）；$k$ 为每例激活 expert 数，本文全部实验合计激活 4 个——flat 层 $k=4$，两级 hierarchical 每层 $k=2$；$W_g,W_{\text{noise}}\in\mathbb{R}^{d\times n}$ 是门控的两份权重；训练侧 $b$ 为每设备 batch 例数、$P$ 为设备数（论文记 $d$，为免与模型维冲突改记 $P$）。

### Stage 1 · 门控打分并注入可学习噪声

$$\begin{aligned}
H(x)_i &= (xW_g)_i + \epsilon\cdot\operatorname{Softplus}\!\big((xW_{\text{noise}})_i\big),\quad \epsilon\sim\mathcal{N}(0,1) && \text{// } [d]\times[\bcancel{d},n]\to[n]\ \text{——线性打分加逐分量高斯噪声，噪声标准差经 Softplus 由第二份权重控制}
\end{aligned}$$

### Stage 2 · KeepTopK 稀疏化

$$\begin{aligned}
\operatorname{KeepTopK}(H(x),k)_i &= \begin{cases}H(x)_i & H(x)_i\ \text{属于 } H(x)\text{ 的 top-}k\\ -\infty & \text{否则}\end{cases} && \text{// } [n]\xrightarrow{\text{沿 } n \text{ 维保留 top-}k}[n]\ \text{——维度不变，非 top-}k\text{ 分量置 } -\infty
\end{aligned}$$

### Stage 3 · softmax 归一成稀疏门控

$$\begin{aligned}
G(x) &= \operatorname{Softmax}\!\big(\operatorname{KeepTopK}(H(x),k)\big) && \text{// } [n]\xrightarrow{\text{沿 } n \text{ 维归一}}[n]\ \text{——}-\infty\text{ 位 softmax 后恰为 0：门控稀疏且分量和为 1}
\end{aligned}$$

这种稀疏在门控函数中引入理论上的不连续，实践中未观察到问题；噪声项用于负载均衡（§2.1）[^src-shazeer-2017-sparse-moe]。

### Stage 4 · expert 前向与加权汇合

$$\begin{aligned}
E_i(x) &= \operatorname{ReLU}(xW_{i,1})\,W_{i,2} && \text{// } [d]\times[\bcancel{d},h]\to[h]\times[\bcancel{h},d]\to[d]\ \text{——单隐层 FFN，通道展到 } h\text{ 再收回}\\
y &= \sum_{i:\,G(x)_i>0} G(x)_i\,E_i(x) && \text{// } k\text{ 个 } [d]\xrightarrow{\text{按门控加权求和}}[d]\ \text{——只算被选中的 } k\text{ 个 expert，稀疏省算力即在此}
\end{aligned}$$

LM 与 MT 实验中，MoE 输出经 sigmoid 后接 dropout，层间设残差连接（Appendix C/E）[^src-shazeer-2017-sparse-moe]。

### Stage 5 · 大 batch：展平时间维与混合并行（训练）

$$\begin{aligned}
B_{\text{expert}} &\approx \tfrac{kb}{n}\;\xrightarrow{\;\text{MoE 层合并 } P\ \text{路同步 batch}\;}\;\tfrac{kbP}{n} && \text{// } P\ \text{份 } [b,d]\ \text{按门控分桶，每 expert 得约 } \tfrac{kb}{n}\ \text{例 } [d]\ \text{——稀疏稀释 batch，expert 单副本模型并行把它乘回 } P\ \text{倍}
\end{aligned}$$

分工（§3.1）：标准 LSTM 层与门控网络按常规数据并行复制，各设备同步执行各自 batch；每个 expert 只保留一份，分片驻留固定设备。MoE 层合并 $P$ 份数据后按门控将样本发给对应 expert——同一组设备既是数据并行副本又是模型并行分片，expert batch 由此获得 "factor of $d$ improvement"（其 $d$ 为设备数，即本页 $P$）[^src-shazeer-2017-sparse-moe]。上一层 LSTM 算完全部 $T$ 个时间步后，MoE 对所有位置一次调用（卷积式应用），batch 再乘 $T$；若递归使用 MoE（每步输入依赖上步输出）该技巧失效，可通过重计算式 BPTT 节省激活（§3.1）[^src-shazeer-2017-sparse-moe]。带宽侧（§3.2）：expert 静止、门控参数小，通信内容主要是 expert 的输入与输出；单隐层 FFN 的计算/通信比恰为 $h$，加大 $h$ 或加深即可压过 GPU 的千比一算力带宽比[^src-shazeer-2017-sparse-moe]。

## 负载均衡：importance 损失与 load 损失

门控自增强地收敛到只给固定几个 expert 大权重：被选中者训练更快，因而更易被选中。Eigen et al. 2013 用训练初期的硬约束避开该局部极小，本文改用软约束（§4）[^src-shazeer-2017-sparse-moe]。

**Importance 损失**（式 6–7）：expert 的 importance 定义为批内门控值之和，损失取 importance 向量变异系数的平方乘手工系数：

$$\operatorname{Importance}(X)=\sum_{x\in X}G(x),\qquad L_{\text{importance}}(X)=w_{\text{importance}}\cdot\operatorname{CV}\!\big(\operatorname{Importance}(X)\big)^2$$

其中 $\operatorname{CV}(Y)=\sigma_Y/\mu_Y$ 跨 $n$ 个 expert 计算。它只均衡门控质量而非样本数：一个 expert 可能分得少量大权重样本，另一个分得大量小权重样本，分布式硬件上仍会引发内存与性能问题（§4）[^src-shazeer-2017-sparse-moe]。

**Load 损失**（式 8–11，Appendix A）：被分配样本数是离散量、不可反传，因而借噪声构造光滑估计。$P(x,i)$ 表示固定其他分量的已采噪声、仅对分量 $i$ 重新采样时 $G(x)_i\neq 0$ 的概率：此时 $H(x)_i$ 须超过除自身外的第 $k$ 大值，对高斯噪声积分得

$$P(x,i)=\Phi\!\left(\frac{(xW_g)_i-\operatorname{kth\_excluding}(H(x),k,i)}{\operatorname{Softplus}\big((xW_{\text{noise}})_i\big)}\right),\qquad \operatorname{Load}(X)_i=\sum_{x\in X}P(x,i)$$

$\Phi$ 为标准正态 CDF。负载损失同样取 CV 平方：$L_{\text{load}}(X)=w_{\text{load}}\cdot\operatorname{CV}(\operatorname{Load}(X))^2$。式 4 的噪声在此把「每个 expert 所分样本数」这一离散计数松弛为对 $W_{\text{noise}}$ 可微的连续量[^src-shazeer-2017-sparse-moe]。初始化将 $W_g,W_{\text{noise}}$ 置零，以在无信号、纯噪声的初始阶段保持负载近似均匀；软约束起效需要时间，开局失衡会直接引发 OOM（Appendix A）[^src-shazeer-2017-sparse-moe]。

## 两级 hierarchical MoE（Appendix B）

$n$ 很大时采用两级分层降低分支因子：primary 门控在 $a$ 个组上稀疏加权，每组为自带门控 $G_i$ 的 secondary MoE，共 $ab$ 个 expert，输出

$$y_H=\sum_{i=1}^{a}\sum_{j=1}^{b}G_{\text{primary}}(x)_i\,G_i(x)_j\,E_{i,j}(x)$$

（式 12）。利用率指标随之改写，secondary 负载记为

$$\operatorname{Load}_H(X)_{i,j}=\frac{\operatorname{Load}_{\text{primary}}(X)_i\cdot\operatorname{Load}_i\big(X^{(i)}\big)_j}{|X^{(i)}|}$$

$X^{(i)}$ 是 primary 选了组 $i$ 的样本子集。不直接写 $\operatorname{Load}_i(X^{(i)})_j$ 是因其对 primary 门控没有梯度，乘积形式把 secondary 负载信号传回 primary（式 13–14）。两级已够用，无需更深层级[^src-shazeer-2017-sparse-moe]。

## 实验证据

### 语言建模：1B word（Table 1、Table 7）

架构为两层 LSTM 中夹一层 MoE（Figure 1）。等算力约 8M ops/timestep 扫容量（Figure 2-left、Table 7）：MoE-4（无稀疏，$k=n=4$）45.0、MoE-32 39.7、MoE-256 35.7、MoE-256-h 36.0、MoE-1024-h 34.6、MoE-4096-h 34.1——4096 expert 比同算力 dense 基线（44.7–46.1）test ppl 低约 24%[^src-shazeer-2017-sparse-moe]。

| 模型 | test ppl（10 epochs） | 参数（除 embedding/softmax） | ops/timestep | 训练时间 | TFLOPS/GPU |
|---|---|---|---|---|---|
| 2xLSTM-8192-1024（已发表最优，Jozefowicz 2016） | 34.7 | 151M | 151M | 59 h / 32 K40 | 1.09 |
| MoE-4096-h（低算力档） | 34.1 | 4303M | 8.9M | 15 h / 16 K40 | 0.74 |
| MoE-34M | 31.3 | 4314M | 33.8M | 17 h / 32 K40 | 1.22 |
| MoE-143M | 28.0 | 4371M | 142.7M | 47 h / 32 K40 | 1.56 |

Table 1 中，低算力档以已发表最优约 6% 的运算量（8.9M 对 151M）在 10 epochs 口径下超过它；MoE-143M 比已发表最优低 18%（§C.2；按 28.0/34.7 实算为 19.3%，照录论文口径）[^src-shazeer-2017-sparse-moe]。

### 语言建模：100B word 与 137B 参数（Table 8）

训练单次处理约 1000 亿词、batch 约 250 万词，显存靠 expert 隐层激活不存而反传重算，以及 expert 上 Adam 取 $\beta_1=0$、二阶矩用行/列均值向量外积近似，压至单卡约 10 亿参数（Appendix D）[^src-shazeer-2017-sparse-moe]。

| 模型 | test ppl（1 epoch） | MoE 层参数 | ops/timestep | TFLOPS/GPU |
|---|---|---|---|---|
| 4xLSTM-512（等算力基线） | 47.0 | 8.4M | 8.4M | 1.23 |
| MoE-256-h | 35.3 | 272.9M | 8.4M | 1.11 |
| MoE-4096-h | 30.9 | 4303M | 8.6M | 1.07 |
| MoE-65536-h | 28.9 | 68791M（约 68B） | 9.2M | 0.72 |
| MoE-131072-h | 29.2 | 137578M（约 137B） | 9.7M | 0.30 |

Table 8 中，65536 expert（99.994% 层稀疏）比基线低 39%；131072 expert 回升至 29.2，归因于可能过于稀疏，且该档训练 batch 未随 GPU 数等比放大（64/128 GPU 集群，效率跌至 0.30 TFLOPS/GPU）；10B 词时各档差距更小（Figure 3 顶线），容量收益随数据量增大[^src-shazeer-2017-sparse-moe]。

### 机器翻译（Table 2–5）

单语言对模型改自 GNMT（encoder LSTM 9→3 层、decoder 8→2 层），encoder 第 2/3 层与 decoder 第 1/2 层间各插入一层最多 2048 expert 的 MoE（每 expert 约 2M 参数，合计约 8B，总参 8.7B），计算量 85M ops/timestep 对 GNMT 的 214M[^src-shazeer-2017-sparse-moe]。

| 任务 | MoE BLEU / ppl | GNMT BLEU / ppl | GNMT+RL BLEU | 训练时间 |
|---|---|---|---|---|
| WMT'14 En→Fr newstest2014（Table 2） | 40.56 / 2.63 | 39.22 / 2.79 | 39.92 | 6 天/64 K40 对 6 天/96 K80 |
| WMT'14 En→De newstest2014（Table 3） | 26.03 / 4.64 | 24.91 / 5.25 | 24.66 | 1 天/64 K40 对 1 天/96 K80 |
| Google Production En→Fr（Table 4，test） | 36.57 / 2.69 | 35.56 / 2.87 | — | 1 天/64 K40 对 6 天/96 K80 |

模型不经 RL 精调仍比 GNMT 高 1.34（En→Fr）与 1.12 BLEU（En→De）；在 Production 集上以六分之一训练时间高 1.01 test BLEU（§5.3）[^src-shazeer-2017-sparse-moe]。

多语单模型（Table 5）：12 语言对一个模型，MoE 层为非分级 $n=512$、$k=2$、每 expert 隐层 8192，102M ops/timestep，总参 8.7B 对 GNMT-Multi 的 278M；dev ppl 3.35 对 4.14（论文称低 19%）；12 对里 11 对胜 GNMT-Multi（最大 +5.84，Ko→En）、8 对胜单语 GNMT；En→Ko 低 1.79，论文归因稀有语言对过采样导致的严重过训。训练 12 天/64 K40 对 21 天/96 GPU[^src-shazeer-2017-sparse-moe]。

### 消融：均衡损失权重（Table 6）

MoE-256、1B word、10 epochs，扫 $w_{\text{importance}}$/$w_{\text{load}}$：

| $w_{\text{importance}}$ / $w_{\text{load}}$ | test ppl | CV(Importance) | CV(Load) | max/mean Load |
|---|---|---|---|---|
| 0 / 0 | 39.8 | 3.04 | 3.01 | 17.80 |
| 0.2 / 0 | 35.6 | 0.06 | 0.17 | 1.47 |
| 0 / 0.2 | 35.7 | 0.22 | 0.04 | 1.15 |
| 0.1 / 0.1 | 35.6 | 0.06 | 0.05 | 1.14 |
| 0.01 / 0.01 | 35.7 | 0.48 | 0.11 | 1.37 |
| 1.0 / 1.0 | 35.7 | 0.03 | 0.02 | 1.07 |

两个损失任取其一即可恢复质量（39.8 → 35.6–35.7），且对权重不敏感（0.01 到 1.0 仅差 0.1 ppl）；$w_{\text{load}}$ 越大最重 expert 的负载越低（1.47 → 1.07）。LM 主实验用 0.1/0.1，MT 用 0.01/0.01（Appendix A/C/E）[^src-shazeer-2017-sparse-moe]。论文未提供 $k$ 扫描或单独关闭噪声的消融，噪声通过 load 损失（式 9 依赖噪声分布）间接检验。

## 专家专业化与论文自述的边界

WMT'14 En→Fr 编码器 2048 expert 上按 $G(x)_i$ 降序观察各 expert 的 batch 样本，expert 381 聚在 innovation/researchers 语境，expert 752 聚在 "plays a core/critical/leading…"，expert 2004 聚在 fast/rapidly/swiftly 类速率表述，专家按句法与语义自发分工（Table 9）[^src-shazeer-2017-sparse-moe]。MoE 卷积式插在堆叠 LSTM 层之间，每个时间步独立路由；其收益依赖大训练集，1B word 上 MoE 层参数超过约 1B 后收益递减，促使转向 100B 语料（§5.2）[^src-shazeer-2017-sparse-moe]。针对当时基础设施对「每 expert 恰好等 batch」更快的特点，strictly balanced gating（Appendix F）训练时用 batchwise mask（每 expert 跨 batch 保 top-$m$，$m=k|X|/n$），推理改用学到的 per-expert 阈值 $T$（式 19–20）；该方案仅部分 MT 实验使用，多语模型仍用 noisy top-k。attention 函数（Appendix G）把 GNMT 的 $A_{GNMT}(x_i,y_j)=\sum_d V_d\tanh((x_iU)_d+(y_jW)_d)$ 改成 $\sum_d V_d\tanh((x_iU)_d)\tanh((y_jW)_d)$，多源多目标步可合并成矩阵乘，质量差异很小。hierarchical 配置（Appendix C/D）在 LM 低算力系列设一级分支 16（等于 GPU 数），100B 系列设 32–256，每层 $k=2$、合计 4 expert。

## 谱系

向前：门控混合专家由 Jacobs et al.（1991）作为顶层模型引入；Eigen et al.（2013）首次将带独立门控的 MoE 作为深度模型内部组件堆叠；本文在此基础上引入稀疏门控与噪声，首次给出条件计算的大规模实证（§1.3、§6）[^src-shazeer-2017-sparse-moe]。向后：noisy top-k 门控与均衡损失是后续稀疏 MoE 的共同起点；top-1 硬路由与不同形式辅助损失的分化，以及在时序与 LLM 中的当代形态，见 [[mixture-of-experts]] 与 [[moe-in-time-series-and-spatio-temporal]]。

[^src-shazeer-2017-sparse-moe]: [[source-shazeer-2017-sparse-moe]]
