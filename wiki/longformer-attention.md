---
title: "Longformer Attention"
type: technique
tags:
  - attention
  - sparse-attention
  - long-context
  - transformer
  - pretraining
created: 2026-10-26
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Longformer Attention

标准自注意力打分矩阵为 $n\times n$，时间与显存均按 $O(n^2)$ 增长，长文档任务（分类、QA、共指消解）只能截断、切块或改用两阶段架构。Longformer（AI2，Beltagy, Peters & Cohan，arXiv:2004.05150v2）按「注意力模式」稀疏化打分矩阵，显式指定互相关注的位置对，将打分矩阵从稠密降为带状加少量全满行列，作为预训练 Transformer 中标准自注意力的 drop-in replacement，支持序列长度至 4,096（字符级 LM 实验到 23,040，评测序列 32,256）[^src-beltagy-2020-longformer]。

## 三种注意力与混合矩阵结构

论文将全注意力拆成三种模式的混合（§3.1，Fig 2b–d）[^src-beltagy-2020-longformer]：

滑动窗口（sliding window）中，每个 token 仅关注固定窗口 $w$ 内的位置，两侧各 $\lfloor w/2\rfloor$ 个，单层复杂度为 $O(n\times w)$，随 $n$ 线性；$\ell$ 层堆叠后顶层感受野为 $\ell\times w$（各层 $w$ 固定）。空洞滑动窗口（dilated sliding window）在窗口内以步长 $d$ 跳采，key 取 $|i-j|=kd$ 的位置，key 数不变而扩大跨度，各层 $d,w$ 固定时感受野为 $\ell\times d\times w$。global attention 允许少数任务指定位置与全序列对称互看（操作对称：global token 读一切，也被一切读），具体位置由任务给定：分类用 `[CLS]`，QA 用全部 question tokens，LED 摘要用 encoder 首个 `<s>`；因 global token 数与 $n$ 无关且远小于 $n$，混合模式整体仍为 $O(n)$。

两个自由度：**每层可用不同 $w$**（字符级 LM 低层用小窗、高层用大窗）；**每个头可用不同 dilation 配置**——部分头不空洞专注局部、部分头空洞负责长程，优于全部头同配置（§3.1、§4.1）。global 位置使用**独立投影** $Q_g,K_g,V_g$（窗口注意力用 $Q_s,K_s,V_s$），$Q_g,K_g,V_g$ 用 $Q_s,K_s,V_s$ 的值初始化；WikiHop 消融中去掉独立投影 −1.6、投影与 global 一起去掉 −8.3（Δ 相对 4,096 基线 73.8，Table 10），二者对下游性能均属关键（§3.1）。

矩阵形态上，非空洞混合模式为带宽 $w$ 的带状矩阵，加 $g$ 个全满行与全满列（global 位置）；空洞时带内非零对角改为 $|i-j|=kd$ 各条。

## 数据流与维度

> **维度符号**：$n$ 序列长度；$w$ 窗口宽（每侧 $\lfloor w/2\rfloor$）；$d$ dilation 步长；$G\subset[n]$ global 位置集合，$g=|G|$；$d_k$ 单头通道维；$\ell$ 层数。每行窗口 key 数严格为 $2\lfloor w/2\rfloor+1$（含自身），与论文口径一致记作 $w$，$O(nw)$ 记号不受影响。

### Stage 1 · 写出依赖集合

$$\begin{aligned}
N_i^{\text{sw}} &= \{\,j:\ |i-j|\leq \lfloor w/2\rfloor\,\} && \text{// } [n]\ \text{个 token 各挂 } w\ \text{个 key，打分总数 } nw\ \text{——}O(nw)\text{ 复杂度的来源}\\
N_i^{\text{dil}} &= \{\,j:\ |i-j|=kd,\ k=0,1,\dots,\lfloor w/2\rfloor\,\} && \text{// } |N_i^{\text{dil}}|=|N_i^{\text{sw}}|\ \text{——key 数不变（含自身项 } k{=}0\text{），相邻 key 间隔拉开 } d \text{ 倍}\\
N_i^{\text{mix}} &= N_i^{\text{sw}}\cup G && \text{// } |N_i^{\text{mix}}|\leq w+g\ \text{——普通 token 的行：窗口加 global 锚点}\\
N_g &= [n]\quad(g\in G) && \text{// } [n]\ \text{——global 行读全序列，同时第 } g\ \text{列被全序列读}
\end{aligned}$$

按 token 类型划分依赖集合：$g\in G$ 的行是 $[n]$（读一切）、列也是 $[n]$（被一切读）；普通 token $i$ 的行是 $N_i^{\text{sw}}\cup G$、列是 $\{\,j:\ |i-j|\leq\lfloor w/2\rfloor\,\}\cup G$。其中仅 global 位置与全序列双向互看，普通 token 之间仍只有窗口边。打分沿用 softmax 内积形式（论文式 (1)），窗口打分与 global 打分分别由两套投影给出，掩码换成 $N_i^{\text{mix}}$ [^src-beltagy-2020-longformer]。

### Stage 2 · 连通层数（覆盖论证）

前提：各层 $w$、$d$ 固定，无 global token，无因果掩码。

$$\begin{aligned}
N_i^{\text{dil}} &\subseteq i+d\mathbb{Z} && \text{// 位移全为 } d\ \text{的倍数——纯空洞堆叠下 } i\ \text{只能到达 } j\equiv i \pmod{d}\\
t_{\min}(i,j) &= \big\lceil |i-j| \,\big/\,\big(d\lfloor w/2\rfloor\big)\big\rceil && \text{// 前提 } j\equiv i\ (\mathrm{mod}\ d)\ \text{——同余类内连通所需最少层数}\\
\ell\, d\, w &\geq n && \text{// 论文的覆盖口径——顶层感受野宽度 } \ell d w\ \text{盖住全序列}
\end{aligned}$$

第二行推导显示：空洞边位移均为 $d$ 的倍数，层数再多同余类也不混合；跨同余类连通至少需要一条 $d=1$ 路径。论文配置与此一致：低层不用空洞以保留局部容量，高层只对 2 个头加递增 dilation（§4.1），细粒度连通由无空洞层提供。$d=1$ 时区间连续，$t_{\min}=\lceil|i-j|/\lfloor w/2\rfloor\rceil$，顶层感受野 $\ell w$ 符合论文 §3.1 计算。字符级 LM 的自回归模式下 kernel 只允许每个 token 查看前面的窗口（附录 A），可达区间相应改为向后单侧 [^src-beltagy-2020-longformer]。

### Stage 3 · 双投影打分

$$\begin{aligned}
S^{\text{sw}} &= Q_s K_s^{\top}/\sqrt{d_k} && \text{// } [n,\bcancel{d_k}]\times[\bcancel{d_k},n]\to[n,n]\ \text{——只计算带内 } nw\ \text{个打分，其余不落盘}\\
S^{g} &= Q_g K_g^{\top}/\sqrt{d_k} && \text{// } [g,\bcancel{d_k}]\times[\bcancel{d_k},n]\to[g,n]\ \text{——global 行对全序列打分，} g\ \text{与 } n\ \text{无关}\\
A\,V &= \operatorname{softmax}\!\big(\mathrm{mask}(S^{\text{sw}},S^{g})\big)V && \text{// } [n,n]\xrightarrow{\text{沿 key 维归一化}} [n,n]\ \text{——掩码外置 } -\infty\text{，对 } [n,d_k]\ \text{加权}
\end{aligned}$$

### Stage 4 · chunk 打分的显存（预训练/微调实现）

$$\begin{aligned}
Q,K &\xrightarrow{\ \text{按 } w\ \text{切块、块间重叠 } w/2\ } \tfrac{2n}{w}\ \text{块} && \text{// } [n,d_k]\to[\tfrac{2n}{w},w,d_k]\ \text{——每个带段由一个 } w\times w\ \text{块对覆盖}\\
\tilde{S} &= Q_{\text{blk}}K_{\text{blk}}^{\top} && \text{// } [\tfrac{2n}{w},w,\bcancel{d_k}]\times[\tfrac{2n}{w},\bcancel{d_k},w]\to[\tfrac{2n}{w},w,w]\ \text{——一次批量矩阵乘}\\
\tilde{S} &\xrightarrow{\ \text{带外置 } -\infty\ } S && \text{// } \tfrac{2n}{w}\times w^2=2nw=O(nw)\ \text{——对比全矩阵 } n^2\text{；块重叠引入约 } 2\times\ \text{冗余打分（论文口径：显存多耗 } 2\times\text{）}
\end{aligned}$$

全注意力显存瓶颈在于把 $n\times n$ 打分矩阵（及 softmax 结果）完整存入显存；chunk 实现只保留带内打分，显存降至 $O(nw)$，Fig 1 显示其显存随序列长度线性增长，而全注意力在长序列上 OOM。论文对比三种实现（§3.2、附录 A）：**loop** 逐对角线计算，省显存但运行极慢，仅用于测试；**chunk** 仅支持非空洞，因计算部分零值比最优实现多耗约 2 倍显存，但走单次 PyTorch 矩阵乘，用于预训练与微调；**cuda** 为 TVM 定制 kernel，功能完整（支持空洞与自回归模式），最省显存且速度与高度优化的全注意力相当，用于字符级实验。注意力内部须用 fp32，fp16 会因数值溢出产生 NaN（附录 A）[^src-beltagy-2020-longformer]。

## 证据

字符级语言建模（text8/enwik8，各 100M 字符、90/5/5 切分；分阶段训练设 5 个 phase，序列长 2,048 翻倍至 23,040，窗口同倍增、学习率减半，Table 12）[^src-beltagy-2020-longformer]：

| 实验 | 数字 | 出处 |
|---|---|---|
| 小模型 text8 test BPC | 1.10（dev 1.04，41M 参数），作者报告新 SOTA | Table 2 |
| 小模型 enwik8 test BPC | 1.00（dev 1.02），作者报告新 SOTA | Table 2 |
| 大模型 enwik8 test BPC | 0.99（102M），匹配 Sparse Transformer（$\approx$100M, 0.99），参数量大其两倍以上的 Adaptive Span（209M, 0.98）与 Compressive（277M, 0.97）略优 | Table 3 |
| 窗口配置消融（dev BPC） | 递增 $w$（32→512）1.21 优于固定 $w{=}230$ 的 1.23 与递减（512→32）的 1.24 | Table 4 上 |
| dilation 消融（dev BPC） | 2 头加 dilation 1.20 优于不加的 1.21 | Table 4 下 |

Table 4 在 phase-1 配置、150K 步、小模型、text8 上测试，脚注 4 指出 150K 步的排序未必等于训练到底的排序。

Masked LM 预训练（从 RoBERTa checkpoint 继续，滑窗 $w{=}512$ 与 RoBERTa 同算力，位置嵌入扩至 4,096 并**复制** RoBERTa 的 512 个嵌入初始化；65K 步、序列 4,096、batch 64；语料 Table 13）[^src-beltagy-2020-longformer]：

| 配置 | base / large BPC（训练语料 dev） | 出处 |
|---|---|---|
| RoBERTa（seqlen 512） | 1.846 / 1.496 | Table 5 |
| Longformer（4,096，随机初始化位置嵌入） | 10.299 / 8.738 | Table 5 |
| + 复制位置嵌入 | 1.957 / 1.597 | Table 5 |
| + 2K 步 / + 65K 步继续预训练 | 1.753→1.705 / 1.414→1.358 | Table 5 |
| 只训新位置嵌入（冻结 RoBERTa） | 1.850 / 1.504 | Table 5 |

预训练加入 dilation 效果反而变差；脚注 6 推测其与 RoBERTa 权重不兼容，从头重训才可能受益。

下游六个数据集上下文平均 300–6,589 wordpieces，95 分位最高 17,126，均超 512 限制（Table 6）。Longformer-base 对 RoBERTa-base 的 dev 结果（Table 7）：WikiHop 75.0/72.4（acc）、TriviaQA 75.2/74.3（F1）、HotpotQA 64.4/63.5（joint F1）、OntoNotes 78.6/78.4、IMDB 95.7/95.3、Hyperpartisan 94.8/87.4。2020-05 提交时，Longformer-large 在 WikiHop 81.9、TriviaQA 77.3 超当时榜单 SOTA（78.3、73.3），HotpotQA 73.2 落后 HGN 一分（Table 8、Table 9；脚注 9 指出 BigBird 后来以 16 倍预训练算力刷新，作者列为混杂因素）。WikiHop 消融相对 Longformer 4,096 的 73.8（Table 10，Δ）：15 epochs 75.0（+1.2）；seqlen 512 且换 $n^2$ 注意力（即配置成 RoBERTa 形态）71.7（−2.1）；seqlen 2,048 73.1（−0.7）；无 MLM 预训练 73.2（−0.6）；无独立投影 72.2（−1.6）；无投影且无 global 65.5（−8.3）；只训新位置嵌入 73.5（−0.3）；RoBERTa-base 本身（seqlen 512）72.4（−1.4）。配置成 RoBERTa（512、$n^2$）时 71.7 略低于 RoBERTa-base 本身的 72.4，作者以此论证增益不来自额外预训练。LED 采用 encoder local+global 与 decoder 全注意力，从 BART 初始化并将位置嵌入复制扩至 16K；在 arXiv 摘要上 16,384 输入的 R-1/R-2/R-L 为 46.63/19.62/41.83，同为 4,096 输入时 44.40/17.94/39.76 对 BigBird 的 46.63/19.02/41.77（Table 11）；Fig 3 显示验证集输入从 1K 到 16K，R-1 35.21→46.23、R-2 11.54→19.62。

## 与相邻稀疏注意力工作的区分

[[sparse-transformer-strided-fixed]] 中 Sparse Transformer（Child et al. 2019）的 strided/fixed 模式服务于自回归生成；其中 8×8 块空洞滑窗（BlockSparse）与 Longformer 最接近，但 Longformer 面向迁移学习并引入任务驱动的 global attention（§2）。[[flash-attention]] 属于精确注意力的 IO 优化，不改变计算依赖；Longformer 改动模式本身，两条路线正交。[[pyraformer]] 是金字塔多分辨率图上的 $O(L)$ 时序注意力，依赖由同尺度相邻与父子节点构成，并非滑窗加空洞；[[patchstg-sparse-attention-form]] 将静态结构稀疏与固定掩码用于时空双轴，同属 static structured sparsity。[[native-sparse-attention]] 与 [[moba-attention]] 作为长上下文后续，分别采用三支路（压缩/选择/滑窗）与 KV 块级 MoE 门控，滑窗支路即该模式的自回归形态；[[logsparse-self-attention]] 与 [[deepseek-sparse-attention]] 属同题不同解的时序与索引路线；[[minimax-sparse-attention]] 则把选择交给学习的内容依赖索引，与固定掩码路线相对。

## 边界

global 位置由任务指定，是向注意力加入归纳偏置的手工通道，并非模型学得（§3.1）。空洞与 RoBERTa 预训练权重不兼容（Table 5 相关脚注 6）；chunk 实现不支持空洞，CUDA kernel 依赖 TVM 编译，注意力限 fp32（附录 A）。实际增益取决于任务是否确实需要长程依赖：TriviaQA/IMDB/OntoNotes 增益小，因局部上下文常已足够、或辅助监督已定位相关段（§6.4）；OntoNotes 平均文档仅 506 wordpieces（Table 6）。排行榜数字为提交时点口径，后续被 BigBird 等以更大预训练算力刷新（其脚注 9）。

[^src-beltagy-2020-longformer]: [[source-beltagy-2020-longformer]]
