---
title: "Grouped-Query Attention (GQA)"
type: technique
tags:
  - attention
  - kv-cache
  - efficient-inference
  - llm
  - multi-query
created: 2026-10-26
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Grouped-Query Attention（GQA）

将 $H$ 个 query head 分成 $G$ 组，每组共享一份 K/V 投影：$G=H$ 退化为多头注意力（MHA），$G=1$ 即多查询注意力（MQA）[^src-ainslie-2023-gqa]。MQA 将 K/V 头从 $H$ 压至 1，削减了 KV cache 容量与带宽；但模型越大头数越多，削减幅度越大。GQA 则将带宽与容量的下降比例固定在 $G/H$，不随模型规模漂移[^src-ainslie-2023-gqa]。

与之配套的是 uptraining：从已有 MHA checkpoint 出发，将 K/V 投影按组 mean pooling 后继续预训练 $\alpha$ 比例的原预训练步数，即可得到 MQA/GQA 模型，$\alpha=0.05$ 时约 600 TPUv3 chip-days[^src-ainslie-2023-gqa]。

## 数据流与维度变换

**维度符号**：$H$ 为 query head 数；$G$ 为组数兼 K/V 头数；$d_k$、$d_v$ 为单头 K/V 维；$d_{model}$ 为模型维；$N$ 为单次前向 query 位置数；$s$ 为已缓存前序长度，解码时 $s$ 逐步加一；$b$ 为每元素缓存字节数。张量形状记作 $[\text{行}, \text{列}]$；论文按「每组一个 K/V 头」表述，组内 query head 到组的映射与批量维展开标注 [INFERENCE]。

### Stage 1 · 分组投影出 Q、K、V

$Q$ 保留 $H$ 份，K/V 仅设 $G$ 份；组 $g$ 覆盖 $|\mathcal{G}_g| = H/G$ 个 query head。论文仅说明「把 query head 分成 $G$ 组」，未规定 head 到组的映射，按连续分组处理 [INFERENCE]。

$$\begin{aligned}
Q &= X W_Q && \text{// } [N, \bcancel{d_{model}}]\times[\bcancel{d_{model}}, H d_k]\to[N, H d_k]\ \text{——每个 query head 一份，与 MHA 相同}\\
K_g &= X W_K^{(g)} && \text{// } [N, \bcancel{d_{model}}]\times[\bcancel{d_{model}}, d_k]\ \text{只做 } G \text{ 次}\to[N, d_k]\ \text{——K 头由 } H \text{ 份降到 } G \text{ 份，逐组各一份}\\
V_g &= X W_V^{(g)} && \text{// } [N, \bcancel{d_{model}}]\times[\bcancel{d_{model}}, d_v]\ \text{只做 } G \text{ 次}\to[N, d_v]\ \text{——V 头同降}
\end{aligned}$$

### Stage 2 · 组内各 query head 独立打分

$H$ 个 query head 分别与所在组的 K 计算内积，打分矩阵仍为 $H$ 个：

$$\begin{aligned}
S_h &= Q_h K_{g(h)}^{\top}\big/\sqrt{d_k} && \text{// } [N, \bcancel{d_k}]\times[\bcancel{d_k}, s]\to[N, s]\ \text{——消去 } d_k\text{，第 } h \text{ 头对全部 } s \text{ 个前序位置的打分}
\end{aligned}$$

组内共享 K 的内容：$H/G$ 个 head 共用同一份 $K_g$ 但对应各自的 $Q_h$，注意力权重互不相同，打分计算量与 MHA 一致。

### Stage 3 · 组内独立 softmax 与加权求和

$$\begin{aligned}
A_h &= \operatorname{Softmax}(S_h) && \text{// } [N, s]\xrightarrow{\text{沿 } s \text{ 维逐行 softmax}}[N, s]\ \text{——逐 head 归一化，组内不共享}\\
O_h &= A_h V_{g(h)} && \text{// } [N, \bcancel{s}]\times[\bcancel{s}, d_v]\to[N, d_v]\ \text{——沿 } s \text{ 维加权求和，读的是组内共用的 } V_g
\end{aligned}$$

### Stage 4 · 拼头与输出投影

$$\begin{aligned}
O &= [O_1;\dots;O_H]\, W_O && \text{// } [N, \bcancel{H d_v}]\times[\bcancel{H d_v}, d_{model}]\to[N, d_{model}]\ \text{——输出投影保持 } H d_v \text{ 宽，与 MHA 同}
\end{aligned}$$

### Stage 5 · 增量解码：每步读取的 K/V 字节

解码时每步只有 1 个新 query，每层 K/V 缓存为 $[G, s, d_k]$ 与 $[G, s, d_v]$，需整份读出：

$$\begin{aligned}
B_{\text{GQA}} &= \left(G d_k + G d_v\right) s\, b && \text{// } [G, s, d_k]\text{ 与 } [G, s, d_v] \xrightarrow{\text{沿 } G \text{ 维累加字节}} G(d_k{+}d_v)\,s\,b\ \text{——随 } G \text{ 线性}\\
B_{\text{MHA}} &= \left(H d_k + H d_v\right) s\, b, \quad B_{\text{MQA}} = \left(d_k + d_v\right) s\, b && \text{// } G=H \text{ 与 } G=1 \text{ 的两端}\\
\frac{B_{\text{GQA}}}{B_{\text{MHA}}} = \frac{G}{H}, \qquad \frac{B_{\text{GQA}}}{B_{\text{MQA}}} &= G && \text{// } \text{读取量比即头数比；MQA 相对 MHA 的 } 1/H \text{ 削减被 GQA 放宽到 } G/H
\end{aligned}$$

KV cache 显存占用依同组式子度量（$s$ 取全序列缓存长度而非单步），随头数缩放同样为 $G/H$：MQA 相对 MHA 省 $H$ 倍缓存，GQA 省 $H/G$ 倍。同一步注意力算术量不变：$QK^{\top}$ 需 $H d_k s$ 次乘加、$AV$ 需 $H d_v s$ 次，与 $G$ 无关（Stage 2–3 每 head 独立）。算术强度（FLOP/byte）相对 MHA 提高 $H/G$ 倍——GQA 换的是访存，不是算力。K/V 投影参数同比例缩放：每层从 $(H d_k + H d_v)d_{model}$ 降至 $(G d_k + G d_v)d_{model}$，即 K 头投影形状由 $[d_{model}, H d_k]$ 变为 $[d_{model}, G d_k]$，V 头同理。

三档对照：

$G=H$ 时，$B$ 与 MHA 相同，KV cache 与 MHA 同尺寸并退化为 MHA；$G=1$ 时，每步只读 $(d_k+d_v)s b$（即 MQA），K/V 头从 $H$ 压至 1，容量与带宽降幅最大[^src-ainslie-2023-gqa]；$1<G<H$ 时，KV cache 大小、K/V 投影参数量与每步读取量均按 $G/H$ 缩放，注意力计算量与输出投影不变。

规模论证显示：KV cache 随 $d_{model}$ 线性增长，FLOPs 与参数量随 $d_{model}$ 平方增长。大模型带宽开销占比更低，但头数通常随规模增加，MQA 固定压缩在大模型上削减比例过甚；GQA 使削减比例与规模脱钩[^src-ainslie-2023-gqa]。此外，大模型按分片数复制唯一的 K/V 头（Pope et al., 2022），GQA 消除了这种复制浪费[^src-ainslie-2023-gqa]。分片数为 $P$ 时，MQA 的有效读取量回到 $P(d_k+d_v)s b$，$G\ge P$ 的 GQA 各片各持 $G/P$ 组而不必复制，这就是「消掉复制浪费」的量化形式 [INFERENCE]。

GQA 仅作用于 decoder 的 self-attention 与 cross-attention，不改 encoder self-attention：encoder 表示并行计算，访存通常不是瓶颈（§2.2、§3.1）[^src-ainslie-2023-gqa]。

## Uptraining 与 mean-pooling 初始化

从 MHA checkpoint 转 GQA 分两步：换结构、再继续预训练（§2.1）[^src-ainslie-2023-gqa]。

**转换**：组 $g$ 的 K/V 投影取组内各头投影的算术平均。

$$\begin{aligned}
W_K^{(g)} &= \frac{1}{|\mathcal{G}_g|}\sum_{h\in\mathcal{G}_g} W_K^{(h)} && \text{// } [d_{model}, \bcancel{|\mathcal{G}_g| d_k}]\xrightarrow{\text{沿列按组取平均}}[d_{model}, d_k]\ \text{——组 } g \text{ 的 K 头，列块数由 } H \text{ 降到 } G\\
W_V^{(g)} &= \frac{1}{|\mathcal{G}_g|}\sum_{h\in\mathcal{G}_g} W_V^{(h)} && \text{// } [d_{model}, \bcancel{|\mathcal{G}_g| d_v}]\xrightarrow{\text{同样按组取平均}}[d_{model}, d_v]\ \text{——V 头同理}
\end{aligned}$$

由线性可知，$x W_K^{(g)}$ 等于组内各头 $x W_K^{(h)}$ 的均值：新 K 头不引入预训练之外的方向，但组内方向相反的分量互相抵消 [INFERENCE]。

**继续预训练**：$\alpha$ 是继续预训练的步数比例，以原预训练步数为分母，沿用原预训练配方与数据集。$\alpha=0.05$ 约 600 TPUv3 chip-days（§2.1、§3.1）[^src-ainslie-2023-gqa]。

**为什么平均能当初始化**：依据来自经验排序，mean pooling 优于取首头、优于随机初始化，理由是「排序与预训练信息的保留程度一致」（§3.3、Figure 4）[^src-ainslie-2023-gqa]。文中未给误差界，未证明平均为组内聚合的最优形式；转换后即使不继续预训练，GQA 也已有可用质量，MQA 则需 uptraining（§3.3、Figure 5）[^src-ainslie-2023-gqa]。

## 证据

**设置**（§3.1）：T5.1.1 Large 与 XXL，JAX/Flax/Flaxformer 实现，Adafactor 与同 T5 的学习率调度；对照为 MHA 的 Large/XXL 与 $\alpha=0.05$ uptrain 的 XXL MQA 与 GQA-8；任务为五个摘要集（CNN/Daily Mail、arXiv、PubMed、MediaSum、Multi-News）、WMT 2014 英德翻译、TriviaQA 问答，不评 GLUE 一类分类基准（自回归推理在那里不适用）。微调常学习率 0.001、batch 128、dropout 0.1；CNN/DM 与 WMT 输入 512 / 输出 256，其余摘要集 2048 / 512，TriviaQA 2048 / 32；贪心解码。计时为每样本每 TPUv4 芯片耗时（xprof），8 TPU、batch 最大 32/TPU、并行策略按模型单独调优。T5 Large/XXL 的层数、$d_{model}$ 与头数 $H$ 在全文未给。

**Table 1**（dev 集；$T_{\text{infer}}$ 为每样本推理时间，表中列名记 s；Average 为各任务平均）[^src-ainslie-2023-gqa]：

| 模型 | $T_{\text{infer}}$ | Average | CNN R1 | arXiv R1 | PubMed R1 | MediaSum R1 | MultiNews R1 | WMT BLEU | TriviaQA F1 |
|---|---|---|---|---|---|---|---|---|---|
| MHA-Large | 0.37 | 46.0 | 42.9 | 44.6 | 46.2 | 35.5 | 46.6 | 27.7 | 78.2 |
| MHA-XXL | 1.51 | 47.2 | 43.8 | 45.6 | 47.5 | 36.4 | 46.9 | 28.4 | 81.9 |
| MQA-XXL | 0.24 | 46.6 | 43.0 | 45.0 | 46.9 | 36.1 | 46.5 | 28.5 | 81.3 |
| GQA-8-XXL | 0.28 | 47.1 | 43.5 | 45.4 | 47.7 | 36.3 | 47.2 | 28.4 | 81.6 |

读法：GQA-8 与 MHA-XXL 平均差 0.1（47.1 vs 47.2），七个指标列中 PubMed 47.7、MultiNews 47.2 两列更高，CNN/arXiv/TriviaQA 各低 0.2–0.3，WMT 持平；推理时间 0.28 vs 1.51，快 5.4 倍。相对 MQA-XXL 平均高 0.5，时间为 0.28 vs 0.24。在 5% uptraining 下 GQA-8 质量接近 MHA，但平均仍未达到 MHA-XXL。

**Figure 3**：平均性能对平均推理时间，MHA-Large/XXL 与 $\alpha=0.05$ 的 MQA-XXL、GQA-8-XXL 四点；uptrained MQA 相对 MHA 是质量更高且更快的取舍（优于 MHA-Large），GQA 在此之上再取质量、速度贴近 MQA[^src-ainslie-2023-gqa]。

**消融**（§3.3，三任务子样本：CNN/DM、MultiNews、TriviaQA）：

Figure 4 显示 T5-Large 转 MQA、$\alpha=0.05$，三种初始化 mean / first / random 中 mean 最优，纵轴范围 54.4–55.6，具体点值不可读[^src-ainslie-2023-gqa]。Figure 5 中 T5-XXL 的 MQA 与 GQA-8 性能随 $\alpha\in[0,0.1]$ 变化，两者都在 5% 处取得主要收益，10% 后边际递减[^src-ainslie-2023-gqa]。Figure 6 显示 GQA-XXL 每样本时间随组数 $G\in\{1,4,8,16,32,64\}$ 变化（输入 2048 / 输出 512），从 1 到 8 组只添少量开销，越接近 MHA 代价越陡，据此选 $G=8$[^src-ainslie-2023-gqa]。Appendix A 中从零训练的 T5-Large MQA 预训练频繁出现 loss spike，微调到长输入任务即发散；uptrained MQA 更稳但方差仍大，不稳定任务上报三次微调平均；uptrained GQA 表现稳定，MQA 不稳定的根因未追查[^src-ainslie-2023-gqa]。

## 边界

方法仅在 encoder-decoder（T5.1.1）上验证，任务限于翻译、摘要与 TriviaQA；decoder-only 模型没有独立的 self-attention 与 cross-attention，GQA 作用面更宽，论文预期其相对 MQA 优势更强，但未实验[^src-ainslie-2023-gqa]。uptraining 需要原始 MHA checkpoint、原预训练配方与数据集，$\alpha=0.05$ 折合 600 TPUv3 chip-days，缺少原始 checkpoint 或预算则无法走通[^src-ainslie-2023-gqa]。受算力限制，实验未与从零训练的 GQA-XXL 对比，uptraining 相对从头训练的差距未测；摘要质量以 ROUGE 衡量，该指标不覆盖全部质量维度（Limitations）[^src-ainslie-2023-gqa]。组内 query head 如何映射到组以及 head 维在 TPU 上的切分方式，论文未展开。

## 相关页面

[[source-ainslie-2023-gqa]] 为源摘要；[[multi-query-attention]] 是 $G=1$ 的端点，K/V 头退化为单份；[[qk-norm]] 作用于 Q/K 归一化，不改头共享结构；[[kv-cache-compression]] 用于推理期缓存压缩，GQA 改的是头共享结构而非缓存内容；[[native-sparse-attention]] 与 [[deepseek-sparse-attention]] 在 GQA / MQA 分组之上做块与 token 粒稀疏；[[compressed-sparse-attention]] 与 [[heavily-compressed-attention]] 落在 $G=1$ 端点并把序列维再压 $1/4$ 与 $1/128$（DeepSeek-V4）。

[^src-ainslie-2023-gqa]: [[source-ainslie-2023-gqa]]
