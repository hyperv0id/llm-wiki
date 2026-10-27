---
title: "Engram：条件记忆与可扩展查表"
type: technique
tags:
  - conditional-memory
  - n-gram
  - sparse-model
  - lookup
  - mixture-of-experts
  - long-context
  - deepseek
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Engram：条件记忆与可扩展查表

Engram 是 DeepSeek-AI 与北京大学（arXiv:2601.07372v2，2026-07-12 版；arXiv 编号对应 2026-01 首发）提出的条件记忆模块：将后缀 $N$-gram 经 tokenizer 压缩与多头哈希映射到静态嵌入表，按 token ID $O(1)$ 查表，经当前隐藏状态门控后注入 Transformer 残差流[^src-deepseek-2026-engram]。论文将其与 MoE 的条件计算并列为另一条稀疏轴，并以固定参数预算下的分配实验给出 U 形曲线[^src-deepseek-2026-engram]。

## 定义与定位

### 条件记忆

论文将语言建模拆为两类子任务：组合推理依赖深层动态计算；而命名实体、固定搭配等文本多为局部、静态且高度模式化，经典 $N$-gram 即可捕捉[^src-deepseek-2026-engram]。标准 Transformer 缺乏原生查表原语，只能由早期注意力与 FFN 逐层重新计算静态查表[^src-deepseek-2026-engram]。

论文在 §1 对照了两条稀疏轴：条件计算（conditional computation）通过稀疏激活参数处理动态逻辑，条件记忆（conditional memory）通过稀疏查表取回固定知识的静态嵌入[^src-deepseek-2026-engram]。Engram 是后者的具体实例，定义见摘要、§1 与 §2.1：检索阶段将局部上下文映射为静态记忆条目，融合阶段由隐藏状态调制检索结果[^src-deepseek-2026-engram]。

> [!note] 定义出处
> “条件记忆”的散文定义在摘要与 §1 第 3 段；模块的两阶段定义在 §2.1；查表、门控、卷积与残差注入的形式化分别在式 (1)–(5) 与 §2.3–§2.4。

### 与 MoE、注意力稀疏的分工

| 稀疏轴 | 稀疏化的对象 | 选择依据 | 主要省下或换来什么 |
|---|---|---|---|
| MoE 条件计算 | 每 token 激活哪些 FFN 专家 | 运行时隐藏状态经 router 打分 | 总算力不随专家总数线性增长 |
| 注意力稀疏（NSA、DSA） | 每个 query 读取哪些 KV 位置 | query–key 打分或压缩块选择 | 注意力计算与 KV 读取从全量变为子集 |
| Engram 条件记忆 | 每 token 读取哪些记忆表行 | 输入 token ID 的 $N$-gram 哈希 | 静态模式不必用深度重建，表容量可超出 HBM |

MoE 一行按论文 §1、§3 口径[^src-deepseek-2026-engram]；Engram 不改变注意力打分，也不减少 KV cache，而是增加 token ID 到记忆表的查表路径[^src-deepseek-2026-engram]。论文以 complementary axis 与 structurally complementary 描述二者关系，未给出正交性的形式化定义；此处“正交”指稀疏对象不同且功能互补[INFERENCE]。论文未与 [[native-sparse-attention|NSA]]、[[deepseek-sparse-attention|DSA]] 做同台实验；“第三条稀疏轴”来自论文的定位与分配实验，而非与注意力稀疏的对照结果[INFERENCE]。

## 模块与前向数据流

**维度符号**：$T$ 为序列长度（token 数）；$d=2560$ 为模型隐藏维；$N=3$ 为最大 $N$-gram 阶数，实际取 $n=2,\dots,N$；$K=8$ 为每个阶数的哈希头数；$d_{\mathrm{mem}}=1280$ 为拼接后的记忆向量维；$d_h=d_{\mathrm{mem}}/((N-1)K)=80$ 为单头嵌入维；$B=4$ 为 mHC 残差分支数；$M_{n,k}$ 为单个 $(n,k)$ 表的槽位数（取素数）。论文未逐项列出 $d_h$，由式 (2) 的拼接关系与 Table 5 的 $d_{\mathrm{mem}}$、$K$、$N$ 反推[INFERENCE]。

### Stage 1 · 压缩词表并切后缀 $N$-gram

$$\begin{aligned}
x'_t &= P(x_t) && \text{// } [T]\xrightarrow{\text{逐 token 规范映射}}[T]\ \text{——NFKC、小写等等价形式折成同一 canonical ID}\\
G_t &= \left(g_{t,2},\dots,g_{t,N}\right),\quad g_{t,n}=\left(x'_{t-n+1},\dots,x'_t\right) && \text{// } [T]\xrightarrow{\text{每个位置取 } N-1 \text{ 个后缀窗口}}[T,N-1]\ \text{——每格是一个 } n\text{-gram 元组}
\end{aligned}$$

$P:V\to V'$ 是预计算的满射，按规范化后的文本等价性合并原始 token[^src-deepseek-2026-engram]。在 128k 词表上，这一步将有效词表缩小 23.43%（Table 6）；合并最多的五类依次是空白符（163 个原始 token）与 a/o/e/i 的大小写、重音、前导空格变体（54/40/35/30 个）[^src-deepseek-2026-engram]。

> [!note] 这里的“压缩”指词表
> 论文的 compression 是 tokenizer 层面的等价 token 合并，不是 KV cache 压缩；全文也没有对记忆表做量化，未给表项位宽[^src-deepseek-2026-engram]。

### Stage 2 · 多头哈希取行并拼接

$$\begin{aligned}
z_{t,n,k} &= \varphi_{n,k}(g_{t,n}) && \text{// } [T,N-1,K]\xrightarrow{\text{乘法-XOR 哈希}}[T,N-1,K]\ \text{——每个 } n\text{-gram 在每个头得到一个表内下标}\\
e_{t,n,k} &= E_{n,k}[z_{t,n,k}] && \text{// } [T,N-1,K]\xrightarrow{\text{按 } z \text{ 取行}}[T,N-1,K,d_h]\ \text{——只读被索引的 }(N-1)K\text{ 行}\\
e_t &= \operatorname{Concat}_{n=2}^{N}\operatorname{Concat}_{k=1}^{K} e_{t,n,k} && \text{// } [T,(N-1)Kd_h]=[T,d_{\mathrm{mem}}]\ \text{——16 段单头向量拼成一条记忆向量}
\end{aligned}$$

这里没有 query–key 打分，也没有 top-$k$ 选槽：每个 token 固定读 $(N-1)K=16$ 行并全部拼接[^src-deepseek-2026-engram]。$K$ 个头与素数表宽用于降低碰撞；碰撞的 $N$-gram 共享同一行，前向得到同一向量，反向将梯度累加到该行[INFERENCE]。表 $E_{n,k}\in\mathbb{R}^{M_{n,k}\times d_h}$ 的槽位数与 $d_h$ 决定参数量，与 $T$ 无关。

### Stage 3 · 用隐藏状态做门控

先做共享值投影，再为每个 mHC 分支算一个标量门：

$$\begin{aligned}
v_t &= W^V e_t && \text{// } [T,d_{\mathrm{mem}}]\times[d_{\mathrm{mem}},d]\to[T,d]\ \text{——所有分支共用同一份值}\\
k_t^{(b)} &= W^{K(b)} e_t && \text{// } [T,d_{\mathrm{mem}}]\times[d_{\mathrm{mem}},d]\to[T,d]\ \text{——第 } b \text{ 个分支自己的键}\\
\alpha_t^{(b)} &= \sigma\!\left(\frac{\sum_{j=1}^{d}\operatorname{RMSNorm}(h_t^{(b)})_j\operatorname{RMSNorm}(k_t^{(b)})_j}{\sqrt d}\right) && \text{// } [T,\bcancel{d}]\xrightarrow{\text{沿 } d \text{ 求内积}}[T]\ \text{——每 token 每分支一个 }(0,1)\text{ 门}\\
u_t^{(b)} &= \alpha_t^{(b)}\,v_t && \text{// } [T]\odot[T,d]\to[T,d]\ \text{——上下文不匹配时门趋零，压掉碰撞或歧义噪声}
\end{aligned}$$

式 (4) 对 query 与 key 均做 RMSNorm 再点积以稳定梯度[^src-deepseek-2026-engram]。$W^V$ 在所有分支共享，$W^{K(b)}$ 逐分支独立；两类投影融合成一次稠密 FP8 矩阵乘（§2.4）[^src-deepseek-2026-engram]。该融合仅涉及投影算子，记忆表未以 FP8 存储。

### Stage 4 · 深度可分离卷积与残差注入

$$\begin{aligned}
\tilde V^{(b)} &= \operatorname{RMSNorm}(U^{(b)}) && \text{// } [T,d]\to[T,d]\ \text{——先归一化再卷积}\\
Y^{(b)} &= \operatorname{SiLU}\!\left(\operatorname{Conv1D}(\tilde V^{(b)})\right)+\tilde V^{(b)} && \text{// } [T,d]\xrightarrow{\text{depthwise 因果卷积}}[T,d]\ \text{——核宽 } w=4\text{、膨胀 } \delta=N\text{，扩大感受野}\\
H^{(b,\ell)} &\leftarrow H^{(b,\ell)}+Y^{(b)} && \text{// } [T,d]+[T,d]\to[T,d]\ \text{——残差写回，随后才是标准 Attention 与 MoE}
\end{aligned}$$

卷积参数零初始化，训练起点严格保持恒等映射[^src-deepseek-2026-engram]。Engram 仅在特定层插入：27B 模型只放在第 2、15 层，位置同时受建模效果与推理延迟约束[^src-deepseek-2026-engram]。

**训练配置**（Table 5、§4.1）：DeepSeek-v3 tokenizer，词表 129,280；30 个 block，$d=2560$，MLA 32 头，mHC 展开率 4，序列长度 4096，batch 1280，训练 50k 步、262B token；主干用 Muon，Engram 表用 Adam（学习率 $\times5$、weight decay 0），卷积零初始化；Engram 取 $d_{\mathrm{mem}}=1280$、$K=8$、$\{2,3\}$-gram，插在第 2、15 层[^src-deepseek-2026-engram]。

## 查表式、梯度与记账

### 为什么是 $O(1)$

键来自离散 token ID 而非隐藏状态：同一段上下文在训练与推理时得到同一组下标，单 token 查表次数固定为 $(N-1)K$，与表槽位数 $M_{n,k}$ 无关[^src-deepseek-2026-engram]。因此增大表只增加 $P_{\mathrm{tot}}$，不增加每 token 的 FLOPs，这构成了“稀疏参数预算”的前提（§3.1）[^src-deepseek-2026-engram]。

### 梯度怎么走

未使用直通估计（STE），也未重参数化查表：哈希函数 $\varphi_{n,k}$ 是固定的乘法-XOR，不含可学习参数，下标由输入决定[^src-deepseek-2026-engram]。前向按 gather 取行，反向把梯度 scatter 回被选中的行；单步内未被任何 token 选中的行不更新[INFERENCE]。训练时表按 GPU 分片，前向用 All-to-All 收集活跃行、反向用 All-to-All 派发梯度（§2.5）[^src-deepseek-2026-engram]。

表参数用 Adam 优化，学习率为基线的 5 倍、weight decay 为 0；卷积零初始化；主干仍用 Muon（Table 5、§4.1）[^src-deepseek-2026-engram]。门控采用 sigmoid，梯度正常回传；式 (4) 的 RMSNorm 稳定门控梯度，而非离散下标可微化[^src-deepseek-2026-engram]。

### 参数与带宽记账

$$\begin{aligned}
d_h &= \frac{d_{\mathrm{mem}}}{(N-1)K}=\frac{1280}{16}=80 && \text{// 单头宽度由拼接维反推}\\
P_{\mathrm{Engram}} &= L_{\mathrm{eng}}(N-1)K\,M\,d_h && \text{// } L_{\mathrm{eng}}\text{ 为插入层数，}M\text{ 为单表槽数}\\
P_{\mathrm{Engram}}^{27\mathrm{B}} &= 2\times16\times2{,}262{,}400\times80\approx5.79\mathrm{B} && \text{// 论文报 5.7B，差 0.09B}\\
P_{\mathrm{Engram}}^{40\mathrm{B}} &= 2\times16\times7{,}239{,}680\times80\approx18.53\mathrm{B} && \text{// 论文报 18.5B}
\end{aligned}$$

按 Table 5 的 27B 配置，$M=2{,}262{,}400$ 为单个 $(n,k)$ 表的槽数；两层各 16 张表，相乘得 5.79B，比论文所报 5.7B 高 0.09B[INFERENCE]。单 token 每层读 16 行、两层共 32 行，即 $32\times80=2{,}560$ 个标量，与 $M$ 无关[INFERENCE]。5.7B 占 26.7B 总参数约 21.3%，18.5B 占 39.5B 约 46.8%；激活参数始终为 3.8B[^src-deepseek-2026-engram]。

论文未给出记忆表的数值位宽，Table 4 仅按“100B 参数表”测试吞吐[^src-deepseek-2026-engram]。推理时表可整体置于主机 DRAM，按确定性下标提前预取，并按 Zipf 分布构建多级缓存（§2.5）[^src-deepseek-2026-engram]。

## 与 MoE 的组合方式

**计算图上是串联，参数预算上是分配。** Engram 经残差支路写回 $H^{(\ell)}$，之后同一 block 仍走标准 Attention 与 MoE（式 5 之后）[^src-deepseek-2026-engram]，而非与 MoE 输出加权并联。两者共享非激活参数预算：

$$\begin{aligned}
P_{\mathrm{sparse}} &= P_{\mathrm{tot}}-P_{\mathrm{act}} && \text{// 不随每 token 计算增长的“免费”容量}\\
P_{\mathrm{MoE}}^{(\mathrm{sparse})} &= \rho P_{\mathrm{sparse}},\qquad P_{\mathrm{Engram}}=(1-\rho)P_{\mathrm{sparse}} && \text{// } \rho\in[0,1]\text{ 为分给 MoE 专家的比例}
\end{aligned}$$

$\rho=1$ 是纯 MoE；调小 $\rho$ 会减少 routed experts，将腾出的参数分给 Engram 槽位[^src-deepseek-2026-engram]。27B 模型把 routed experts 从 72 降至 55，腾出 5.7B 给记忆 $\rho=74.3\%$；top-6 激活专家数不变，激活参数仍是 3.8B（Table 1、§4.1）[^src-deepseek-2026-engram]。

该组合并非“正交可叠加”：论文在两档计算预算下均得到 U 形曲线，纯 MoE 与纯 Engram 都不如混合[^src-deepseek-2026-engram]。MoE 占满时缺少静态记忆，模型只能用深度重建模式；Engram 占满时失去条件计算，动态推理受损[^src-deepseek-2026-engram]。因此这里的“互补”是功能互补与预算竞争，而非两项收益直接相加。

## 稀疏容量分配

实验固定 $P_{\mathrm{tot}}$ 与 $P_{\mathrm{act}}$，稀疏比 $P_{\mathrm{tot}}/P_{\mathrm{act}}\approx10$，只改 routed experts 数与 Engram 槽位数[^src-deepseek-2026-engram]：

$C=2\times10^{20}$ FLOPs 对应 $P_{\mathrm{tot}}\approx5.7$B、$P_{\mathrm{act}}=568$M，$\rho=1$ 基线 106 个专家；$C=6\times10^{20}$ FLOPs 对应 $P_{\mathrm{tot}}\approx9.9$B、$P_{\mathrm{act}}=993$M，基线 99 个专家。

结果（Figure 3 左）[^src-deepseek-2026-engram]：

两档曲线均呈 U 形。把 MoE 份额降到 $\rho\approx40\%$（5.7B 档 46 个专家、9.9B 档 43 个）时，Engram 仍与纯 MoE 持平。最优在 $\rho\approx75\%$–$80\%$，即约 20%–25% 的稀疏预算给 Engram；固定稀疏度下，该位置在两档间保持稳定。9.9B 档验证损失从 $\rho=100\%$ 的 1.7248 降到最优附近的 1.7109，差 0.0139。

无限内存一侧（Figure 3 右）：固定 3B MoE 主干（$P_{\mathrm{act}}=568$M）、训练 100B token，槽位数从 $2.58\times10^5$ 扫到 $1.0\times10^7$，最多增至约 13B 参数；验证损失随槽位数在 log 空间线性下降[^src-deepseek-2026-engram]。同预算下与 OverEncoding 的直接平均相比，Engram 的曲线更陡[^src-deepseek-2026-engram]。论文仅给出趋势，未提供幂律指数或外推公式。

## 预训练主结果（Table 1）

四个模型同数据课程、同 262B token、激活参数均为 3.8B：Dense-4B、MoE-27B（2 共享 + 72 routed，top-6）、Engram-27B（2 + 55，top-6，5.7B Engram）、Engram-40B（18.5B Engram）[^src-deepseek-2026-engram]。

| Benchmark（Metric） | Shots | Dense-4B | MoE-27B | Engram-27B | Engram-40B |
|---|---:|---:|---:|---:|---:|
| 总参数 | - | 4.1B | 26.7B | 26.7B | 39.5B |
| 激活参数（不含 token embed） | - | 3.8B | 3.8B | 3.8B | 3.8B |
| 训练 token | - | 262B | 262B | 262B | 262B |
| 专家（共享 + routed，top-$k$） | - | - | 2 + 72 (top-6) | 2 + 55 (top-6) | 2 + 55 (top-6) |
| Engram 参数 | - | - | - | 5.7B | 18.5B |
| Pile (loss) | - | 2.091 | 1.960 | 1.950 | 1.942 |
| 验证集 (loss) | - | 1.768 | 1.634 | 1.622 | 1.610 |
| MMLU (Acc.) | 5 | 48.6 | 57.4 | 60.4 | 60.6 |
| MMLU-Redux (Acc.) | 5 | 50.7 | 60.6 | 64.0 | 64.5 |
| MMLU-Pro (Acc.) | 5 | 21.1 | 28.3 | 30.1 | 31.3 |
| CMMLU (Acc.) | 5 | 47.9 | 57.9 | 61.9 | 63.4 |
| C-Eval (Acc.) | 5 | 46.9 | 58.0 | 62.7 | 63.3 |
| AGIEval (Acc.) | 0 | 29.1 | 38.6 | 41.8 | 45.9 |
| ARC-Easy (Acc.) | 25 | 76.8 | 86.5 | 89.0 | 90.1 |
| ARC-Challenge (Acc.) | 25 | 59.3 | 70.1 | 73.8 | 76.4 |
| TriviaQA (EM) | 5 | 33.0 | 48.8 | 50.7 | 51.8 |
| TriviaQA-ZH (EM) | 5 | 62.8 | 74.8 | 76.3 | 77.9 |
| PopQA (EM) | 15 | 15.1 | 19.2 | 19.4 | 21.2 |
| CCPM (Acc.) | 0 | 72.2 | 79.6 | 87.1 | 87.7 |
| BBH (EM) | 3 | 42.8 | 50.9 | 55.9 | 57.5 |
| HellaSwag (Acc.) | 0 | 64.3 | 71.8 | 72.7 | 73.1 |
| PIQA (Acc.) | 0 | 63.8 | 71.9 | 73.5 | 76.5 |
| WinoGrande (Acc.) | 5 | 64.0 | 67.6 | 67.8 | 68.1 |
| DROP (F1) | 1 | 41.6 | 55.7 | 59.0 | 60.7 |
| RACE-Middle (Acc.) | 5 | 72.4 | 80.9 | 82.8 | 83.3 |
| RACE-High (Acc.) | 5 | 66.0 | 75.4 | 78.2 | 79.2 |
| C3 (Acc.) | 0 | 57.7 | 60.1 | 63.6 | 61.8 |
| HumanEval (Pass@1) | 0 | 26.8 | 37.8 | 40.8 | 38.4 |
| MBPP (Pass@1) | 3 | 35.4 | 46.6 | 48.2 | 46.2 |
| CruxEval-i (EM) | 0 | 27.6 | 30.7 | 32.2 | 36.2 |
| CruxEval-o (EM) | 0 | 28.7 | 34.1 | 35.0 | 35.3 |
| GSM8K (EM) | 8 | 35.5 | 58.4 | 60.6 | 62.6 |
| MGSM (EM) | 8 | 27.0 | 46.8 | 49.4 | 52.4 |
| MATH (EM) | 4 | 15.2 | 28.3 | 30.7 | 30.6 |

相比同参同 FLOPs 的 MoE-27B，Engram-27B 在 Table 1 的 27 项下游基准全部更高，两个 loss 更低[^src-deepseek-2026-engram]。增益不只在知识类任务：MMLU +3.0、CMMLU +4.0、MMLU-Pro +1.8；推理与代码数学增益更大：BBH +5.0、ARC-Challenge +3.7、DROP +3.3、HumanEval +3.0、GSM8K +2.2、MATH +2.4[^src-deepseek-2026-engram]。

> [!warning] 摘要与表格的 MMLU 增益不一致
> 摘要与 §1 写 MMLU +3.4，§4.2 与 Table 1 的 57.4→60.4 给出 +3.0。两处并列照录，不替论文选一个数字。

Engram-40B 继续降低 Pile 与验证 loss，并在多数基准上高于 27B，但并未全面占优：MBPP 46.2 低于 MoE-27B 的 46.6，C3 61.8、HumanEval 38.4、MATH 30.6 低于 Engram-27B 的 63.6、40.8、30.7[^src-deepseek-2026-engram]。这源于欠训练：40B 与基线的训练 loss 差距在训练末段仍在扩大[^src-deepseek-2026-engram]。

## 长上下文（Table 2）

预训练后统一做 32k 上下文扩展：YaRN，$s=10$、$\alpha=1$、$\beta=32$、$f=0.707$，5,000 步、30B 长上下文 token[^src-deepseek-2026-engram]。比较四种配置：MoE-27B 50k 步；Engram-27B 的 41k、46k、50k 检查点。46k 被选为 iso-loss 点，其预训练 loss 与 MoE-27B 50k 相同[^src-deepseek-2026-engram]。

| 模型 | LongPPL Book | Paper | Code | L-CoT | NIAH S | MK | MV | MQ | VT | CWE | FWE | QA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MoE-27B (50k, 1.63) | 4.38 | 2.91 | 2.49 | 14.16 | 100.0 | 88.0 | 92.7 | 84.2 | 77.0 | 4.5 | 73.0 | 34.5 |
| Engram-27B (41k, 1.66) | 4.37 | 2.92 | 2.50 | 14.26 | 99.6 | 88.3 | 93.0 | 89.5 | 83.2 | 3.8 | 99.6 | 44.0 |
| Engram-27B (46k, 1.63) | 4.19 | 2.84 | 2.45 | 13.59 | 97.6 | 89.0 | 95.5 | 97.0 | 87.2 | 4.3 | 98.6 | 37.5 |
| Engram-27B (50k, 1.62) | 4.14 | 2.82 | 2.44 | 13.41 | 99.3 | 89.3 | 96.5 | 97.0 | 89.0 | 5.9 | 99.3 | 40.5 |

- 41k 只有 82% 的预训练 FLOPs（41k 对 50k），LongPPL 四项中三项略高于 MoE-27B 50k（Paper 2.92 对 2.91、Code 2.50 对 2.49、L-CoT 14.26 对 14.16），Book 略低（4.37 对 4.38），论文概括为“matches”；RULER 已更高：MQ 89.5 对 84.2、VT 83.2 对 77.0、FWE 99.6 对 73.0[^src-deepseek-2026-engram]。
- iso-loss（46k）下，MQ 97.0 对 84.2、VT 87.2 对 77.0；iso-FLOPs（50k）下进一步到 MQ 97.0、VT 89.0[^src-deepseek-2026-engram]。
- 单针 NIAH 从 100.0 降到 46k 的 97.6，再回到 50k 的 99.3；CWE 一列两侧都低（4.5 对 5.9），论文没有单独解释[^src-deepseek-2026-engram]。

论文把长上下文增益归因于局部依赖被查表接管，注意力容量腾给全局上下文[^src-deepseek-2026-engram]。该论断的证据来自上述对照分析。

## 机制分析

### 实体消解需要多层计算（Table 3）

Table 3 转引 Ghandeharioun et al. (2024) 的 PatchScope 案例：对 “Diana, Princess of Wales”，第 1–2 层只解析到 “Country in the United Kingdom / Wales”，第 3–5 层逐步解析出 “Princess of Wales” 称谓，第 6 层才出现完整实体与生平[^src-deepseek-2026-engram]。论文借此说明，静态实体需经多层 Attention 与 FFN 逐级组合，而查表可以直接替代该过程[^src-deepseek-2026-engram]。

### LogitLens 与 CKA

LogitLens 将每层隐藏状态输入最终 LM head，计算与最终输出分布的 KL。Engram-27B 与 Engram-40B 在各层均更低，早期层差距最大、曲线下降更陡，特征组合完成得更早（Figure 4a）[^src-deepseek-2026-engram]。在 Few-NERD 上，CKA 取命名实体末 token 隐藏状态计算 Engram 层与 MoE 层的表示相似度，再用 top-$k$（$k=5$）加权质心定义软对齐层号 $a_j$（式 9）[^src-deepseek-2026-engram]。热图高相似对角线整体上移，例如 Engram-27B 第 5 层表示最接近 MoE 基线约第 12 层（Figure 4b–c）[^src-deepseek-2026-engram]。论文据此主张 Engram 等价于增加有效深度，早期层不必再做静态重建[^src-deepseek-2026-engram]；证据来自表示相似性与收敛速度，未做因果干预单独隔离“深度增加”[INFERENCE]。

### 抑制 Engram 输出后的保留率（Figure 6）

推理时置零稀疏嵌入输出且保持主干不动，会造成训练–推理不一致；论文重点报告信噪比较高的两类任务[^src-deepseek-2026-engram]：

事实知识只剩 29%–44%，例如 TriviaQA 29%[^src-deepseek-2026-engram]；阅读理解保留 81%–93%，例如 C3 93%[^src-deepseek-2026-engram]。

这组对照说明记忆模块主要承载静态事实知识，上下文 grounding 仍主要依靠主干注意力；此处属于后验抑制，并非正常推理路径[^src-deepseek-2026-engram]。

## 消融与插入位置（Figure 5）

消融主干为 12 层、3B MoE、0.56B 激活，训练 100B token；MoE 基线验证 loss 1.808[^src-deepseek-2026-engram]。参考配置加 1.6B Engram、$\{2,3\}$-gram，插入第 2 与第 6 层，验证 loss 1.768（$\Delta=0.04$）[^src-deepseek-2026-engram]。

将 1.6B 集中为单个模块从第 1 层扫到第 12 层，第 2 层最优（1.770），第 1 层与更深层均变差[^src-deepseek-2026-engram]。其权衡在于：早插能在主干消耗深度前接管局部模式，但早期隐藏状态尚未聚合足够全局上下文，门控更粗；晚插门控更准但卸载效果减弱[^src-deepseek-2026-engram]。把同样 1.6B 预算拆到第 2、6 层，效果优于单层 1.770，达到 1.768，且分层插入有利于铺开预取窗口[^src-deepseek-2026-engram]。组件消融中，去掉多分支逐支路融合、上下文门控、tokenizer 压缩导致的回归最大；去掉轻量 depthwise 卷积只有边际影响；在固定 1.6B 预算下把容量分给 4-gram 略差，原因在于稀释了更常见的 2/3-gram，而在更大记忆下高阶 $N$-gram 仍有用[^src-deepseek-2026-engram]。

Figure 5 仅给出曲线与标记，正文未提供逐项消融的数值差[^src-deepseek-2026-engram]。

## 推理系统（Table 4）

实验基于 nano-vLLM 与 H800，测试 512 条序列，长度 $\mathrm{Uniform}(100,1024)$；100B 参数的 Engram 层整体放在主机 DRAM，插入第二个 Transformer block，按确定性下标异步预取，与第一个 block 的计算重叠[^src-deepseek-2026-engram]。

| 主干 | 配置 | 吞吐（tok/s） |
|---|---|---:|
| Dense-4B | 基线 | 9,031.62 |
| Dense-4B | + 100B Engram（CPU offload） | 8,858.28 |
| Dense-8B | 基线 | 6,315.52 |
| Dense-8B | + 100B Engram（CPU offload） | 6,140.02 |

吞吐损失分别为 1.92% 与 2.78%，论文概括为最高 2.8%[^src-deepseek-2026-engram]。每步通信量随激活槽位数而非表大小增长；实验让所有读取均走 PCIe 且不使用 HBM 缓存，以此作为保守下界[^src-deepseek-2026-engram]。该实验基于 dense 主干，而非 MoE 与 Engram 联合服务。

## 与同期查表路线

论文从两点区分自身与 $N$-gram 嵌入/大表路线（§7）[^src-deepseek-2026-engram]：

评测口径上，SCONE 面向推理且带额外模块与训练 FLOPs，不满足严格 iso-compute；OverEncoding 在稀疏 MoE 主干上即使不控参数也无明显收益；Engram 则在严格同参同 FLOPs 下与 MoE 对比，并给出分配律[^src-deepseek-2026-engram]。系统位置上，已有做法多把嵌入放在第 0 层，访存与计算串行；Engram 插到更深层，用前序层计算掩盖预取，并利用 $N$-gram 的 Zipf 分布做多级缓存[^src-deepseek-2026-engram]。

更早的谱系包括 FastText 式子词嵌入、N-Grammer、BLT 的哈希 $N$-gram，以及 PKM/PEER 一类参数化记忆网络。论文将自身与它们的区别归结为两处：严格同参同 FLOPs 的评测口径，以及把嵌入放到更深层、用前序层计算掩盖预取的系统位置（§7）[^src-deepseek-2026-engram]。

## 边界

规模只到 27B/40B 总参、3.8B 激活与 262B token；40B 未严格全面超过 27B，论文推测欠训练[^src-deepseek-2026-engram]。插入位置只在 12 层 3B 主干、100B token 上扫描；27B 固定第 2、15 层，缺少更大模型的位置扫描[^src-deepseek-2026-engram]。实验只系统测 $\{2,3\}$-gram；4-gram 仅在 1.6B 固定预算下略差，更大记忆预算下的高阶 $N$-gram 未验证[^src-deepseek-2026-engram]。长上下文只扩展到 32k，缺少 128k 或更长序列的结果[^src-deepseek-2026-engram]。文中未给出哈希碰撞率、碰撞对性能影响、表项数值精度或量化实验；Table 4 的 100B 表实验也未报告表的数据类型[^src-deepseek-2026-engram]。评测缺少训练吞吐与显存数字；推理吞吐只在 Dense-4B/8B 上测，不是 MoE 主干与 Engram 同时服务的场景[^src-deepseek-2026-engram]。U 形分配只在两个计算预算上测，未拟合函数形式；最优 $\rho\approx75\%$–$80\%$ 是在固定稀疏比约 10 下的结论[^src-deepseek-2026-engram]。方法未与 NSA/DSA 等注意力稀疏方法同台；“第三条稀疏轴”是定位与分配实验，不是跨轴对照实验[^src-deepseek-2026-engram]。全部实验均为预训练与上下文扩展评测，缺少下游指令微调、RLHF 或 agentic 任务结果[^src-deepseek-2026-engram]。Table 3 为转引；LogitLens/CKA 属表示层面的相关性证据，论文未做因果干预[^src-deepseek-2026-engram]。

## 相关页面

[[mixture-of-experts|MoE]] 对应条件计算一侧的稀疏轴，Engram 与它共享 $P_{\mathrm{sparse}}$ 预算；[[deepseekmoe|DeepSeekMoE]] 则是 27B 主干的专家结构（2 共享 + routed，top-6）。[[native-sparse-attention|NSA]]、[[deepseek-sparse-attention|DSA]] 属于稀疏注意力一侧，稀疏对象是 query 读取的 KV 子集，与 Engram 的查表路径分属不同层。[[multi-head-latent-attention|MLA]]、[[manifold-constrained-hyper-connections|mHC]] 是 27B 主干使用的注意力与多分支残差结构。[[source-2512-02556-deepseek-v3-2|DeepSeek-V3.2]] 同属 DeepSeek 稀疏谱系，但稀疏的是注意力。

[^src-deepseek-2026-engram]: [[source-deepseek-2026-engram]]
