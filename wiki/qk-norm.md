---
title: "QK-Norm（Query-Key Normalization）"
type: technique
tags:
  - transformer
  - attention
  - normalization
  - cosine-similarity
  - temperature
created: 2026-10-26
last_updated: 2026-10-26
source_count: 3
confidence: medium
status: active
---

# QK-Norm（Query-Key Normalization）

QK-Norm 在自注意力内部对 query 与 key 分别做 ℓ2 归一化，把打分从内积转为余弦相似度，再乘可学习标量把打分放回可用尺度[^src-henry-2020-qk-norm]。它与 [[scaling-factor-sqrt-dk|$1/\sqrt{d_k}$ 缩放]] 作用于同一位置——打分矩阵进入 softmax 之前——区别在于尺度不再是 $d$ 的确定性函数，而是训练出来的一个数。

## 机制

### 数据流与维度变换

**维度符号**：$N$ 序列长度（token 数）；$d_{\text{model}}$ 残差流宽度；$h$ 头数，单头通道维 $d=d_{\text{model}}/h$；$L$ 是 $g_0$ 用到的长度统计量，取训练集源侧与目标侧全部序列长度的 97.5 分位。公式按单头书写，$h$ 个头并行、互不共享参数。

### Stage 1 · 投影出每头的 query 与 key

$$\begin{aligned}
q^{(t)}_i &= x_i W^{Q}_t && \text{// } [1, \bcancel{d_{\text{model}}}]\times[\bcancel{d_{\text{model}}}, d]\to[1, d]\ \text{——token } i \text{ 在第 } t \text{ 头得到 } d \text{ 维 query}\\
k^{(t)}_j &= x_j W^{K}_t && \text{// } [1, \bcancel{d_{\text{model}}}]\times[\bcancel{d_{\text{model}}}, d]\to[1, d]\ \text{——同一 token 作 key，投影矩阵独立，不共享 } W^{Q}_t
\end{aligned}$$

### Stage 2 · 逐 token、逐头做 ℓ2 归一化

$$\begin{aligned}
\hat q^{(t)}_i &= q^{(t)}_i \big/ \lVert q^{(t)}_i\rVert_2 && \text{// } [N, d] \xrightarrow{\text{沿 } d \text{ 维逐行归一}} [N, d]\ \text{——每行落到单位球面，模长信息被丢弃}\\
\hat k^{(t)}_j &= k^{(t)}_j \big/ \lVert k^{(t)}_j\rVert_2 && \text{// } [N, d] \xrightarrow{\text{沿 } d \text{ 维逐行归一}} [N, d]\ \text{——只归一 Q 与 K，V 保持原尺度}
\end{aligned}$$

### Stage 3 · 乘一个可学习标量

$$\begin{aligned}
g_0 &= \log_2\left(L^2-L\right) && \text{// } \text{标量，无维度；} L \text{ 取训练集源/目标序列长度的 97.5 分位}\\
g\,\hat q^{(t)\top}_i \hat k^{(t)}_j &= g\cos\left(q^{(t)}_i, k^{(t)}_j\right) && \text{// } [\bcancel{d}]\times[\bcancel{d}]\to[1]\ \text{——打分从内积变成 } g \text{ 倍余弦相似度}
\end{aligned}$$

### Stage 4 · 点积矩阵、softmax 与加权求和

$$\begin{aligned}
S^{(t)} &= g\,\hat Q^{(t)}\hat K^{(t)\top} && \text{// } [N, \bcancel{d}]\times[\bcancel{d}, N]\to[N, N]\ \text{——每格是一对 token 的余弦相似度乘 } g\\
A^{(t)} &= \operatorname{softmax}\!\left(S^{(t)}\right) && \text{// } [N, N] \xrightarrow{\text{沿 key 维逐行 softmax}} [N, N]\ \text{——第 } i \text{ 行是 token } i \text{ 在 } N \text{ 个位置上的分配}\\
O^{(t)} &= A^{(t)}V^{(t)} && \text{// } [N, \bcancel{N}]\times[\bcancel{N}, d]\to[N, d]\ \text{——沿 key 维加权求和，位置轴被消去}\\
O &= \operatorname{Concat}_t\!\left(O^{(t)}\right)W^{O} && \text{// } h\times[N, d]\to[N, hd] \xrightarrow{\times W^{O}} [N, d_{\text{model}}]\ \text{——拼头换轴后投回残差流}
\end{aligned}$$

改动仅在打分矩阵进入 softmax 之前：论文式 (4) 的 $\operatorname{softmax}(QK^{\top}/\sqrt{d})V$ 换成式 (5) 的 $\operatorname{softmax}(g\hat Q\hat K^{\top})V$，$A\to O$ 与多头拼接保持不变[^src-henry-2020-qk-norm]。

### 与 LayerNorm 的分工

QK-Norm 与 LayerNorm 并用，不替代 LayerNorm[^src-henry-2020-qk-norm]。LayerNorm 作用于子层输入，重新缩放并中心化激活分布，控制整条 $[N, d_{\text{model}}]$ 残差流的统计量[^src-henry-2020-qk-norm]；QK-Norm 作用于投影后，每条 $[d]$ 各自归一，不碰 V、不碰残差流。两者正交：LayerNorm 之后 q、k 的模长仍由 $W^{Q}$、$W^{K}$ 的输出尺度决定，QK-Norm 消除了这一自由度[INFERENCE]。

### 与「除以 $\sqrt{d}$」的关系

$1/\sqrt{d}$ 基于「各分量独立、零均值、单位方差时点积方差为 $d$」的假设，用于将方差压回 1[^src-attention-1706-03762]。QK-Norm 从三处替换了该固定常数：

1. 归一化的对象从方差变成上界。余弦打分被硬性限制在 $[-1,1]$，与 $d$ 无关，不必按 $d$ 重新标定[^src-henry-2020-qk-norm]。
2. 缩放方向相反。$\sqrt{d}$ 做除法，$g$ 做乘法——归一化之后打分反而偏小，需要放大[^src-henry-2020-qk-norm]。
3. 常数变成参数。$g$ 随训练更新，初值由数据长度决定，而非由 $d$ 决定[^src-henry-2020-qk-norm]。

ScaleNorm 是这条路上的中间形态：切头前沿 embedding 维 ℓ2 归一化 Q、K、V，再乘一个可学习标量，初值为 $1/\sqrt{d}$，$d$ 为 embedding 维[^src-henry-2020-qk-norm]。QK-Norm 把归一化挪到切头之后、只作用于 Q 与 K，初值也从 $1/\sqrt{d}$ 换成 $\log_2(L^2-L)$。

另一条更窄的路线只归一化 K：[[key-normalization|键归一化]]保留 $\lVert q_i\rVert$ 控制每个查询的锐度，归一化后注意力排名完全由余弦相似度决定[^src-quest]。QK-Norm 同时归一化 Q 与 K，将锐度收归单一全局标量 $g$。

## 关键推导

### 余弦打分为什么有界

点积无界使得微小的相对差异足以压制其余信号，限制了注意力头能学到的模式复杂度；反之，余弦相似度受限于 $[-1,1]$，作为 softmax 输入时差值过小，模型无法有效忽略无关连接，因此不除以 $\sqrt{d}$ 而是乘一个可学习参数[^src-henry-2020-qk-norm]。

代数补充（论文未展开）[INFERENCE]：归一化后 $\lVert\hat q\rVert_2=\lVert\hat k\rVert_2=1$，由 Cauchy–Schwarz

$$\left|\hat q^{\top}\hat k\right|\le \lVert\hat q\rVert_2\lVert\hat k\rVert_2 = 1$$

于是 $\hat Q\hat K^{\top}$ 各元素落在 $[-1,1]$，乘 $g$ 后 $S$ 落在 $[-g,g]$，行内最大差为 $2g$——与 $d$ 无关，也与 $W^{Q}$、$W^{K}$ 的输出尺度、初始化方式无关。对照未缩放的 $QK^{\top}$：分量独立时其方差为 $d$，logit 跨度随 $\sqrt{d}$ 变宽[^src-attention-1706-03762]。$1/\sqrt{d}$ 与 $g$ 因此不是同一量：前者将方差固定为 1，后者约束取值范围上界。

行内最大权重的上界同样不含 $d$，仅含 $g$ 与行内比较数 $N-1$，取行长为 $L$ 即：

$$p_{\max}\le \frac{1}{1+(L-1)e^{-2g}}$$

$g$ 越大上界越贴近 1（允许 one-hot），越小越强制分散。代入 $g_0=\log_2(L^2-L)$ 得 $(L-1)e^{-2g_0}\approx L^{1-4/\ln 2}\approx L^{-4.77}$，上界接近 1，$g_0$ 将上限设在接近 one-hot 处[INFERENCE]。初始化时 $q$、$k$ 尚未分化，各向同性随机向量的余弦集中在 0 附近（标准差约 $1/\sqrt{d}$），实际 logit 跨度约为 $g_0/\sqrt{d}$ 量级，远小于 $2g_0$，注意力近乎均匀；训练把余弦推向 $\pm1$ 之后，$g_0$ 才提供足够宽的可用范围[INFERENCE]。

有界的是 logit 取值而非注意力集中程度：余弦本身仍可逼近 $\pm1$，此时 softmax 依旧接近 one-hot。QK-Norm 消除打分幅度无界导致的饱和，不消除分布过于集中[INFERENCE]。

### $g_0=\log_2(L^2-L)$ 的来源

论文将该初值作为经验法则（rule of thumb），未给推导，指明其取决于序列长度即 $QK^{\top}$ 的元素个数，并按不同分位在 en→vi 上定出 $L$[^src-henry-2020-qk-norm]。若要求行内最弱连接的相对权重 $e^{-2g}$ 降到 $1/(L^2-L)$ 以下（即低于 query-key 对总数的倒数），临界值为 $g^{\star}=\frac12\ln(L^2-L)$；论文取 $\log_2(L^2-L)=(2/\ln 2)\,g^{\star}$，约为该临界值的 2.9 倍，偏向「允许锐化」而非「强制分散」[INFERENCE]。取 97.5 分位而非最大长度，可避免个别超长序列把 $g_0$ 抬得过大[INFERENCE]；Table 5 在 en→vi 上对比 75–99 分位与最大长度，97.5 分位得分最高（33.24，最大长度 33.10）[^src-henry-2020-qk-norm]。

### 参数形态的后果

$g$ 是单一标量，所有头、所有 token、所有位置共用；论文未做逐头或逐 token 的 $g$ 对照。余弦对 $q$、$k$ 的正标量缩放不变，模长归一化后不可恢复，因此 QK-Norm 用全局可学标量替代了模长原本携带的自由度[INFERENCE]。

## 证据

任务包含 5 个低资源翻译方向：4 个取自 TED Talks 语料（Qi et al., 2018）——ar→en、sk→en、gl→en、en→he，1 个取自 IWSLT'15——en→vi；各方向仅 10k–214k 例句、0.37M–8.09M token（Table 1），增益均来自该小样本条件。实现沿用 Nguyen & Salazar (2019) 仓库，PRE-NORM 与 FIX-NORM 均在位，8,000 步线性 warmup 取得最好结果[^src-henry-2020-qk-norm]。

Table 1（数据与训练规模）：

| 方向 | 例句数 | 源+目标 token | 参数量 | 训练时长 (h) | 开发集 BLEU | GPU | $L$ |
|---|---|---|---|---|---|---|---|
| gl→en | 10k | 0.37M | 31,051,880 | 6 | 23.45 | T4 | 79 |
| sk→en | 61k | 2.32M | 48,356,907 | 11 | 31.34 | T4 | 75 |
| en→vi | 133k | 5.99M | 48,431,538 | 19 | 28.77 | T4 | 72 |
| en→he | 212k | 7.88M | 48,401,538 | 38 | 31.16 | T4 | 72 |
| ar→en | 214k | 8.09M | 48,499,512 | 26 | 37.94 | P100 | 75 |

Table 2（测试集 BLEU，Moses 脚本口径，bootstrap 重采样 $p<0.01$）：

| 方向 | Nguyen & Salazar (2019) | QKNORM + LAYERNORM | 差值 |
|---|---|---|---|
| en→vi | 32.79 | 33.24 | +0.45 |
| ar→en | 36.09 | 36.75 | +0.66 |
| en→he | 28.28 | 28.96 | +0.68 |
| gl→en | 22.01 | 24.21 | +2.20 |
| sk→en | 32.58 | 33.23 | +0.65 |
| 平均 | — | — | +0.928 |

Table 3（去词元化后 SacreBLEU 复评）：

| 方向 | Nguyen & Salazar (2019) | QKNORM + LAYERNORM | 差值 |
|---|---|---|---|
| en→vi | 32.41 | 33.18 | +0.77 |
| ar→en | 36.09 | 36.75 | +0.66 |
| en→he | 28.28 | 28.96 | +0.68 |
| gl→en | 22.01 | 24.21 | +2.20 |
| sk→en | 32.58 | 33.22 | +0.64 |
| 平均 | — | — | +0.99 |

Table 2 报告平均 +0.928；Table 3 的 +0.99 由逐列差值算出，论文未报告该平均。两表 en→vi 基线不同：Table 2 的 32.79 取自 Nguyen & Salazar (2019) 报告值，Table 3 的 32.41 为论文复现所得最好 SacreBLEU 成绩[^src-henry-2020-qk-norm]。增益集中在 gl→en（+2.20），其余四向为 0.45–0.68；去掉 gl→en 后其余四向按 Table 2 算为 +0.61。Table 1 仅给出本方法各方向的单卡训练时长（6–38 小时），未给基线时长与参数量，无法逐项对比训练开销[^src-henry-2020-qk-norm]。

Figure 1–2 的独立演示采用 The Annotated Transformer 实现，在 IWSLT 2016 de→en 上训练 10 epoch，基线 19.4 BLEU、QKNORM 24.33 BLEU（SacreBLEU），热图显示 QKNORM 注意力更分散；论文注明该对照「显然不系统」，仅提供直觉[^src-henry-2020-qk-norm]。

外部对照中，en→vi 的 33.18 比 BPE-dropout (Provilkov et al., 2020) 低 0.09 BLEU，而 BPE-dropout 相对 vanilla BPE 提升 1.49[^src-henry-2020-qk-norm]。

附录 A–C 的三组实验均只在 IWSLT'15 en→vi 上完成。Table 4 扫头数 2–32，8 头最高（33.24），32 头仍有 32.30；Table 5 扫 $g_0$ 所用长度分位 75–99 与最大长度，97.5 分位 33.24 高于 99 分位的 32.64 与最大长度的 33.10；Table 6 在 QK-Norm 内部逐项移除组件：去掉可学习标量 $g$ 降至 24.53（完整方法 33.24），去掉 LayerNorm 为 31.56，去掉 FIX-NORM 为 32.63，同时去掉 FIX-NORM 与 PRE-NORM 为 32.20，将 V 也一并 ℓ2 归一化为 32.34[^src-henry-2020-qk-norm]。

## 边界

验证范围仅覆盖 5 个低资源翻译方向、单卡 6–38 小时与 31M–48M 参数，没有语言建模、句法任务或高资源设置的实验；高资源、深度与头数的最优组合、warmup 调度及与 BPE-dropout 的结合均列为后续探索，头数只在 en→vi 上扫过一次（Table 4）[^src-henry-2020-qk-norm]。对照模型与本方法存在两处差异：基线用 ScaleNorm 与缩放点积注意力，本方法用 vanilla LayerNorm 与 QK-Norm。Table 2/3 的 0.928 与 0.99 同时包含这两项改动；Table 6 仅在 QK-Norm 内部逐项移除组件，未设置「基线保留 ScaleNorm、只把打分换成余弦」的对照，两处改动的各自贡献并未拆开[^src-henry-2020-qk-norm]。作者将「确定它究竟在哪起作用、为什么起作用」列为待解问题，热图仅作直觉演示，分散化解释以 conjecture 给出[^src-henry-2020-qk-norm]。论文将 QK-Norm 定位为提高翻译成绩且不牺牲表达能力，依据仅为 5 个方向的 BLEU 与 en→vi 上头数 2–32 的扫描（最高 33.24、最低 32.30），未直接测量表达能力[^src-henry-2020-qk-norm]。论文未报告训练中的 logit 幅度、注意力熵或梯度范数；QK-Norm 仅将每对打分的取值限制在 $[-g,g]$，属于「把尺度常数换成学习式标量」的一个实例，缺少通用稳定性证据，以范数增长为诊断对象的场景见 [[attention-logit-explosion]]。

## 相关页面

[[scaling-factor-sqrt-dk]] 包含 $1/\sqrt{d_k}$ 的方差推导与软饱和论证，QK-Norm 将固定常数换为学习标量；[[key-normalization]] 仅归一化 K，保留 $\lVert q_i\rVert$ 作每查询锐度；[[attention-logit-explosion]] 针对范数无界增长导致的 logit 爆炸，QK-Norm 仅从打分取值上封顶；[[attention-temperature-scaling]] 同属 logits 的乘性缩放，$t$ 按上下文扩展比例手算，$g$ 由训练得到；[[pre-ln-transformer]] 则并用 PRE-NORM 与 QK-Norm。

## 引用

[^src-henry-2020-qk-norm]: arXiv:2010.04245v1 全文（Findings of EMNLP 2020），PDF 归档 `raw/query-key-normalization-for-transformers-2010-04245.pdf`。机制与本页同源，摘要见 [[source-henry-2020-qk-norm]]。
[^src-attention-1706-03762]: [[source-attention-is-all-you-need-1706-03762]]
[^src-quest]: [[source-quest]]
