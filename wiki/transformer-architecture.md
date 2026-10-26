---
title: "Transformer 架构"
type: technique
tags:
  - transformer
  - attention
  - encoder-decoder
  - machine-translation
  - architecture
created: 2026-10-26
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# Transformer 架构

Transformer 用纯注意力替换循环与卷积：编码器、解码器各 6 层，每层由多头注意力与逐位置前馈网络两个子层构成，每个子层接残差连接与层归一化，位置信息以正弦函数加到嵌入上（§3.1、§3.5）[^src-attention-1706-03762]。base 配置 $d_{\text{model}}=512$、$d_{ff}=2048$、$h=8$、$d_k=d_v=64$、dropout 0.1、65M 参数；big 配置 $d_{\text{model}}=1024$、$d_{ff}=4096$、$h=16$、dropout 0.3、213M 参数（Table 3 首行与末行）[^src-attention-1706-03762]。

## 问题定位

循环模型沿符号位置分解计算，$h_t=f(h_{t-1},x_t)$，样本内无法并行，长序列批处理受显存限制（§1）[^src-attention-1706-03762]。卷积路线（ConvS2S、ByteNet）可并行，但任意两位置间所需操作数随距离线性或对数增长，远距离依赖更难学习（§2）[^src-attention-1706-03762]。

Transformer 将任意两位置间的路径长度缩至常数：单个自注意力层直接连接所有位置，代价是注意力加权平均降低了有效分辨率，论文用多头注意力补偿这一损失（§2）[^src-attention-1706-03762]。

## 数据流与维度变换

**维度符号**：$N$ 输入 token 数；$M$ 输出 token 数；$d_{\text{model}}$ 残差流宽度；$h$ 注意力头数；$d_k=d_v=d_{\text{model}}/h$ 单头通道维；$d_{ff}$ 前馈内层宽度；$|\mathcal V|$ 词表大小。全页统一行向量约定：一个 token 占一行，形状记作 $[\text{行},\text{列}]$，权重矩阵右乘，打分矩阵行列分别对应 query 与 key。

### Stage 0 · 嵌入与位置相加

$$\begin{aligned}
X &= \operatorname{Embed}(x_{1:N}) && \text{// } [N] \xrightarrow{\text{按词表查表并乘 } \sqrt{d_{\text{model}}}} [N, d_{\text{model}}]\ \text{——每行一个输入 token 的嵌入}\\
H^{(0)} &= X + PE_{1:N} && \text{// } [N, d_{\text{model}}] + [N, d_{\text{model}}] \to [N, d_{\text{model}}]\ \text{——逐元素相加，位置信息不占额外宽度}\\
Z &= \operatorname{Encoder}\!\left(H^{(0)}\right) && \text{// } [N, d_{\text{model}}] \to [N, d_{\text{model}}]\ \text{——6 层后的记忆，行仍对应输入位置}
\end{aligned}$$

位置向量见 [[sinusoidal-positional-encoding]]；编码器与解码器底部各加一次，层内不再注入（§3.5）[^src-attention-1706-03762]。

### Stage 1 · 单头投影

$$\begin{aligned}
Q^{(j)} &= H W^{Q}_{j} && \text{// } [N, \bcancel{d_{\text{model}}}]\times[\bcancel{d_{\text{model}}}, d_k]\to[N, d_k]\ \text{——第 } j \text{ 个头把残差流投到自己的 } d_k \text{ 子空间，消去 } d_{\text{model}}\\
K^{(j)} &= H W^{K}_{j} && \text{// } [N, \bcancel{d_{\text{model}}}]\times[\bcancel{d_{\text{model}}}, d_k]\to[N, d_k]\ \text{——key 子空间，消去 } d_{\text{model}}\text{，行是位置}\\
V^{(j)} &= H W^{V}_{j} && \text{// } [N, \bcancel{d_{\text{model}}}]\times[\bcancel{d_{\text{model}}}, d_v]\to[N, d_v]\ \text{——value 子空间，消去 } d_{\text{model}}\text{，行是位置}
\end{aligned}$$

$h$ 个头各有独立投影：$W^{Q}_j,W^{K}_j\in\mathbb{R}^{d_{\text{model}}\times d_k}$，$W^{V}_j\in\mathbb{R}^{d_{\text{model}}\times d_v}$（§3.2.2）[^src-attention-1706-03762]。

### Stage 2 · 打分、掩码、归一化、加权

$$\begin{aligned}
S^{(j)} &= Q^{(j)}{K^{(j)}}^{\top}\big/\sqrt{d_k} && \text{// } [N, \bcancel{d_k}]\times[\bcancel{d_k}, N]\to[N, N]\ \text{——第 } i \text{ 行是位置 } i \text{ 对全部 } N \text{ 个位置的打分，消去 } d_k\\
\tilde S^{(j)} &= S^{(j)} + M && \text{// } [N, N] + [N, N] \to [N, N]\ \text{——解码器自注意力令 } M_{ij}=-\infty\ (j>i)\text{、其余为 } 0\text{；编码器取 } M=0\\
A^{(j)} &= \operatorname{softmax}\!\left(\tilde S^{(j)}\right) && \text{// } [N, N] \xrightarrow{\text{沿列（key 维）逐行归一化}} [N, N]\ \text{——每行是位置 } i \text{ 的注意力分布，行和为 } 1\\
O^{(j)} &= A^{(j)} V^{(j)} && \text{// } [N, \bcancel{N}]\times[\bcancel{N}, d_v]\to[N, d_v]\ \text{——沿 key 维加权求和，消去 } N
\end{aligned}$$

掩码作用于 softmax 输入而非输出，因此被屏蔽位置的概率严格为 0（§3.2.3）[^src-attention-1706-03762]。

### Stage 3 · 多头拼接

$$\begin{aligned}
O &= \operatorname{Concat}\!\left(O^{(1)},\dots,O^{(h)}\right)W^{O} && \text{// } [N, \bcancel{h d_v}]\times[\bcancel{h d_v}, d_{\text{model}}]\to[N, d_{\text{model}}]\ \text{——} h \text{ 个 } d_v \text{ 子空间并列拼回残差流宽度}
\end{aligned}$$

$h d_v=d_{\text{model}}$，因而多头的总算力与全维单头相当（§3.2.2）[^src-attention-1706-03762]。多头让不同子空间在不同位置各自关注，单头平均则会削弱这一点（§3.2.2）[^src-attention-1706-03762]。

### Stage 4 · post-LN 残差与前馈

$$\begin{aligned}
H &\leftarrow \operatorname{LayerNorm}\!\left(H + O\right) && \text{// } [N, d_{\text{model}}]\xrightarrow{\text{沿 } d_{\text{model}} \text{ 逐行求均值、方差}}[N, d_{\text{model}}]\ \text{——先加残差再归一化（post-LN）}\\
U &= \operatorname{ReLU}\!\left(H W_1 + b_1\right) && \text{// } [N, \bcancel{d_{\text{model}}}]\times[\bcancel{d_{\text{model}}}, d_{ff}]\to[N, d_{ff}]\ \text{——逐行同一组参数升到 } d_{ff}\\
F &= U W_2 + b_2 && \text{// } [N, \bcancel{d_{ff}}]\times[\bcancel{d_{ff}}, d_{\text{model}}]\to[N, d_{\text{model}}]\ \text{——降回残差流宽度，逐位置独立}\\
H &\leftarrow \operatorname{LayerNorm}\!\left(H + F\right) && \text{// } [N, d_{\text{model}}]\to[N, d_{\text{model}}]\ \text{——第二个子层同构：残差在前、归一化在后}
\end{aligned}$$

前馈网络等价于两次核宽 1 的卷积，各位置共享参数、跨层不共享（§3.3）[^src-attention-1706-03762]。子层输出统一写作 $\operatorname{LayerNorm}(x+\operatorname{Sublayer}(x))$，所有子层与嵌入层输出均保持 $d_{\text{model}}$，残差才能直接相加（§3.1）[^src-attention-1706-03762]。

### Stage 5 · 解码器第一子层：带因果掩码的自注意力

$$\begin{aligned}
S^{dec} &= Q^{dec}{K^{dec}}^{\top}\big/\sqrt{d_k} && \text{// } [M, \bcancel{d_k}]\times[\bcancel{d_k}, M]\to[M, M]\ \text{——行列都是输出位置，位置 } i \text{ 只能看 } \le i\\
A^{dec} &= \operatorname{softmax}\!\left(S^{dec} + M_{\text{causal}}\right) && \text{// } [M, M] \xrightarrow{\text{沿列（key 维）逐行归一化}} [M, M]\ \text{——上三角为 } -\infty\text{，保证自回归性}\\
O^{dec} &= A^{dec}V^{dec} && \text{// } [M, \bcancel{M}]\times[\bcancel{M}, d_v]\to[M, d_v]\ \text{——软对齐到已生成的历史位置}
\end{aligned}$$

输出嵌入右移一位，与掩码共同保证位置 $i$ 的预测只依赖 $<i$ 的已知输出（§3.1）[^src-attention-1706-03762]。

### Stage 6 · 解码器插入的第三子层：cross-attention

$$\begin{aligned}
Q^{dec} &= H^{dec}W^{Q} && \text{// } [M, \bcancel{d_{\text{model}}}]\times[\bcancel{d_{\text{model}}}, d_k]\to[M, d_k]\ \text{——query 来自解码器当前层}\\
K^{enc} &= ZW^{K} && \text{// } [N, \bcancel{d_{\text{model}}}]\times[\bcancel{d_{\text{model}}}, d_k]\to[N, d_k]\ \text{——key 来自编码器输出，行是输入位置}\\
V^{enc} &= ZW^{V} && \text{// } [N, \bcancel{d_{\text{model}}}]\times[\bcancel{d_{\text{model}}}, d_v]\to[N, d_v]\ \text{——value 来自编码器输出，消去 } d_{\text{model}}\\
S^{cross} &= Q^{dec}{K^{enc}}^{\top}\big/\sqrt{d_k} && \text{// } [M, \bcancel{d_k}]\times[\bcancel{d_k}, N]\to[M, N]\ \text{——两条轴长度不同：行是输出位置、列是输入位置}\\
O^{cross} &= \operatorname{softmax}\!\left(S^{cross}\right)V^{enc} && \text{// } [M, \bcancel{N}]\times[\bcancel{N}, d_v]\to[M, d_v]\ \text{——每个解码位置对全部输入位置加权，不加因果掩码}
\end{aligned}$$

cross-attention 是解码器的第三个子层，每个输出位置可访问全部输入位置（§3.2.3）[^src-attention-1706-03762]。

### Stage 7 · 输出分布

$$\begin{aligned}
P &= \operatorname{softmax}\!\left(H^{dec}W^{\top}\right) && \text{// } [M, \bcancel{d_{\text{model}}}]\times[\bcancel{d_{\text{model}}}, |\mathcal V|]\to[M, |\mathcal V|]\ \text{——行是位置、列是下一 token 概率}
\end{aligned}$$

$W$ 与两层嵌入共享，嵌入侧乘 $\sqrt{d_{\text{model}}}$（§3.4）[^src-attention-1706-03762]。

### Stage 8 · 训练路径与推理路径

训练采用教师强制：整条 $M$ 个位置一次前向，以掩码保证并发计算不泄漏未来；损失为逐位置交叉熵，标签平滑 $\epsilon_{ls}=0.1$；残差 dropout 作用于各子层输出加回残差之前，以及嵌入与位置编码之和（§5.4）[^src-attention-1706-03762]。

推理采用自回归：从起始符起每步生成一个 token，将新位置接至 $H^{dec}$ 末尾重算解码器；编码器输出 $Z$ 一次算完、跨步复用。论文采用 beam 4、长度惩罚 $\alpha=0.6$、最大输出长度取输入长度 +50；base 模型平均最后 5 个 checkpoint，big 模型平均最后 20 个（§6.1）[^src-attention-1706-03762]。

## 关键推导：$1/\sqrt{d_k}$ 的方差条件

论文的论证以脚注 4 的前提给出：$q$、$k$ 各分量是独立随机变量，均值 0、方差 1（§3.2.1 脚注 4）[^src-attention-1706-03762]。

$$\begin{aligned}
\operatorname{Var}(q\cdot k) &= \sum_{i=1}^{d_k}\operatorname{Var}(q_i k_i) && \text{// 各乘积项独立，方差逐项相加，共 } d_k \text{ 项}\\
&= \sum_{i=1}^{d_k}\left(\mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] - \left(\mathbb{E}[q_i]\mathbb{E}[k_i]\right)^2\right) && \text{// } q_i,k_i \text{ 独立，交叉矩分解；零均值把第二项归零}\\
&= \sum_{i=1}^{d_k} 1\cdot 1 = d_k && \text{// 单位方差代入，得到点积方差 } = d_k\text{，标准差 } = \sqrt{d_k}
\end{aligned}$$

于是 $\operatorname{Var}\!\left(q\cdot k/\sqrt{d_k}\right)=1$：缩放后的打分尺度与 $d_k$ 无关。不缩放时打分幅度随 $\sqrt{d_k}$ 增长，softmax 被推入梯度极小的区域（§3.2.1）[^src-attention-1706-03762]。该推导给出充分条件：在「独立、零均值、单位方差」输入假设下点积必然膨胀；论文不预设训练后的 $Q,K$ 始终满足该假设，直接对全部配置统一除以 $\sqrt{d_k}$。

同节还给出对照：$d_k$ 小时加法注意力与点积注意力表现接近，$d_k$ 大时未缩放的加法注意力反超，说明差异来自幅度而非交互函数形式；点积注意力实现上更快、更省空间，因为可直接用矩阵乘法（§3.2.1）[^src-attention-1706-03762]。缩放的失效模式与后续讨论见 [[scaling-factor-sqrt-dk]]。

## 复杂度与路径长度

自注意力每层 $O(n^2\cdot d)$，串行操作数 $O(1)$，任意两位置最大路径长度 $O(1)$；循环层为 $O(n\cdot d^2)$、$O(n)$、$O(n)$；卷积层（核宽 $k$）为 $O(k\cdot n\cdot d^2)$、$O(1)$、$O(\log_k n)$；受限邻域自注意力（邻域 $r$）为 $O(r\cdot n\cdot d)$、$O(1)$、$O(n/r)$（Table 1）[^src-attention-1706-03762]。

按总计算量，$n<d$ 时自注意力快于循环层，当时最优翻译模型的句表示（word-piece、BPE）正属此区间；$n$ 远大于 $d$ 时改用受限自注意力，论文未做该实验（§4）[^src-attention-1706-03762]。

## 证据

### 翻译质量与训练成本（Table 2，newstest2014）

| 模型 | EN-DE BLEU | EN-FR BLEU | 训练 FLOPs (EN-DE) | 出处 |
|---|---|---|---|---|
| Transformer (base) | 27.3 | 38.1 | $3.3\times10^{18}$ | Table 2 |
| Transformer (big) | 28.4 | 41.8 | $2.3\times10^{19}$ | Table 2 |
| GNMT + RL | 24.6 | 39.92 | $2.3\times10^{19}$ | Table 2 |
| ConvS2S | 25.16 | 40.46 | $9.6\times10^{18}$ | Table 2 |
| MoE | 26.03 | 40.56 | $2.0\times10^{19}$ | Table 2 |
| GNMT + RL Ensemble | 26.30 | 41.16 | $1.8\times10^{20}$ | Table 2 |
| ConvS2S Ensemble | 26.36 | 41.29 | $7.7\times10^{19}$ | Table 2 |

big 模型 EN-DE 比当时最好的集成模型高 2.04 BLEU，8 卡 P100 训练 3.5 天；base 模型在 EN-DE 上 12 小时、$3.3\times10^{18}$ FLOPs 即超过所有已发表模型与集成（§6.1）[^src-attention-1706-03762]。EN-FR 的 big 模型 BLEU 在 §6.1 正文为 41.0，Table 2 与摘要为 41.8，两处并存；训练 FLOPs 单列为 Table 2 第二列。

### 架构变体（Table 3，EN-DE newstest2013 dev，除注明外均为 base 配置，beam search、无 checkpoint 平均）

| 变体 | dev BLEU | 参数 | 出处 |
|---|---|---|---|
| base（$h=8$，$d_k=d_v=64$，$d_{\text{model}}=512$，$d_{ff}=2048$，$P_{drop}=0.1$） | 25.8 | 65M | Table 3 |
| $h=1$（$d_k=d_v=512$） | 24.9 | — | Table 3 (A) |
| $h=32$（$d_k=d_v=16$） | 25.4 | — | Table 3 (A) |
| $d_{\text{model}}=1024$，$d_k=d_v=128$ | 26.0 | 168M | Table 3 (C) |
| $d_{ff}=4096$ | 26.2 | 90M | Table 3 (C) |
| $P_{drop}=0$ | 24.6 | — | Table 3 (D) |
| $\epsilon_{ls}=0$ | 25.3 | — | Table 3 (D) |
| 学习式位置嵌入替代正弦 | 25.7 | — | Table 3 (E) |
| big（$h=16$，$d_{\text{model}}=1024$，$d_{ff}=4096$，$P_{drop}=0.3$） | 26.4 | 213M | Table 3 |

单头比 $h=8$ 低 0.9 BLEU，头过多也下降（§6.2）[^src-attention-1706-03762]。去掉 dropout 下降 1.2 BLEU，去掉标签平滑下降 0.5；正弦编码与学习式位置嵌入差 0.1，两者效果接近，选正弦旨在向更长序列外推（§3.5、§6.2）[^src-attention-1706-03762]。

### 泛化到成分句法分析（Table 4）

4 层、$d_{\text{model}}=1024$ 的 Transformer 在 WSJ（40K 句）上 F1 达 91.3，半监督（约 17M 句）达 92.7；已发表最优 RNN Grammar 为 93.3（§6.3）[^src-attention-1706-03762]。该任务解码设为 beam 21、$\alpha=0.3$，最大输出长度为输入 +300。

## 训练配置

用 Adam（$\beta_1=0.9$、$\beta_2=0.98$、$\epsilon=10^{-9}$），学习率 $\text{lrate}=d_{\text{model}}^{-0.5}\cdot\min(\text{step}^{-0.5},\ \text{step}\cdot\text{warmup}^{-1.5})$，warmup 4000 步（§5.3）[^src-attention-1706-03762]。EN-DE 约 4.5M 句对，共享 37000 词 BPE 词表；EN-FR 36M 句，用 32000 word-piece 词表；每批约 25000 源 token 与 25000 目标 token（§5.1）[^src-attention-1706-03762]。8 卡 P100 上，base 每步 0.4 秒、训 100K 步；big 每步 1.0 秒、训 300K 步（§5.2）[^src-attention-1706-03762]。

## 边界

每层复杂度 $O(n^2\cdot d)$ 随序列长度增加；受限邻域自注意力（邻域 $r$，路径长度 $O(n/r)$）只作为 future work 提出，未实验（§4）[^src-attention-1706-03762]。注意力对位置加权平均会降低有效分辨率，论文靠多头补偿（§2）[^src-attention-1706-03762]。验证限于 WMT 2014 翻译与 WSJ 成分句法分析；附录只给注意力头的定性检查，未把可解释性作为定量结论（§4、附录）[^src-attention-1706-03762]。

## 相关页面

[[sinusoidal-positional-encoding]] 涵盖位置注入与平移的线性映射，[[scaling-factor-sqrt-dk]] 梳理缩放因子的失效模式与后续讨论，[[interpretable-multi-head-attention]] 记录 TFT 在多头结构上共享 value 并平均注意力图，[[inverted-transformer-architecture]] 包含保留组件、反转注意力作用维度的后续架构，[[probsparse-self-attention]]、[[logsparse-self-attention]] 给出针对 $O(n^2\cdot d)$ 的稀疏化方案，[[source-attention-is-all-you-need-1706-03762]] 为论文摘要页。

---

[^src-attention-1706-03762]: [[source-attention-is-all-you-need-1706-03762]]
