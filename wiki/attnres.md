---
title: "Attention Residuals (AttnRes)"
type: technique
tags:
  - residual-connections
  - depth-attention
  - macro-design
  - large-language-models
  - training-efficiency
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Attention Residuals (AttnRes)

AttnRes（Kimi Team / Moonshot AI，arXiv:2603.15031v1，2026-03）将标准残差连接中全部前序层输出的固定单位权重累加，替换为深度维上的 softmax 注意力：每层仅用一个可学习伪查询向量 $w_l\in\mathbb{R}^d$，对前序各层输出按内容加权检索[^src-moonshot-2026-attnres]。论文将这一步定位为序列维度 linear→softmax 过渡（RNN→Transformer）在深度维度上的重演；它与 [[hyper-connections]]、[[manifold-constrained-hyper-connections]] 的多流递归同属残差宏观设计谱系，走跨层直接访问路线[^src-moonshot-2026-attnres]。

## 问题：固定权重累加与 Pre-LN 稀释

标准残差 $h_l = h_{l-1} + f_{l-1}(h_{l-1})$ 展开为 $h_l = h_1 + \sum_{i=1}^{l-1} f_i(h_i)$，每层接收全部前序输出的等权和。Pre-LN 下隐状态模长随深度以 $O(L)$ 增长，各层相对贡献被逐步稀释，相当比例的层剪掉后损失几乎不变[^src-moonshot-2026-attnres]。由此带来三条局限：不同类型的层（attention 与 MLP）接收同一聚合态，缺乏选择性；聚合中丢失的信息无法在深层选择性找回；后层必须学习更大的输出才能在累加中产生影响，导致训练不稳[^src-moonshot-2026-attnres]。Highway 门控 $h_l = (1-g_l)\odot h_{l-1} + g_l\odot f_{l-1}(h_{l-1})$ 仅放宽相邻两层间的系数，每层仍只能访问紧邻输入 $h_{l-1}$ 这一压缩态[^src-moonshot-2026-attnres]。

## 方法

### Full AttnRes

**维度符号**：输入 $B\times T\times d$；每个 self-attention 或 MLP 各记一层，共 $L$ 层；公式按单 token 书写，$h_l\in\mathbb{R}^d$ 为进入第 $l$ 层的隐状态，$h_1$ 为 token embedding；$N$ 为块数、$S=L/N$ 为块内层数、$B_n$ 为第 $n$ 块的层下标集。

### Stage 1 · 造深度 value 池

$$\begin{aligned}
v_0 &= h_1,\qquad v_i = f_i(h_i)\ (1\le i\le l-1) && \text{// } [d]\ \text{——embedding 与每个已算完层的输出各占一个源，构成深度方向的 KV 池}
\end{aligned}$$

### Stage 2 · 打深度注意力分

$$\begin{aligned}
q_l &= w_l && \text{// } [d]\ \text{——每层一个可学习伪查询，与前向计算解耦}\\
\alpha_{i\to l} &= \frac{\varphi(w_l,\,k_i)}{\sum_{j=0}^{l-1}\varphi(w_l,\,k_j)},\quad \varphi(q,k)=\exp\!\big(q^\top\operatorname{RMSNorm}(k)\big) && \text{// } [d]\times[d]\to\mathbb{R}\xrightarrow{\text{对 }l\text{ 个源做 softmax}}\alpha_{\to l}\in\Delta^{l-1}\ \text{——key 取 value 本身，RMSNorm 防大输出层垄断权重（式 2–3）}
\end{aligned}$$

### Stage 3 · 聚合出层输入

$$\begin{aligned}
h_l &= \sum_{i=0}^{l-1}\alpha_{i\to l}\,v_i && \text{// } [1,\bcancel{l}]\times[\bcancel{l},d]\to[d]\ \text{——消去源维 } l\ \text{——softmax 权重非负且和为 1，}h_l\text{ 是前序输出的凸组合（式 1、4）}
\end{aligned}$$

伪查询必须零初始化：$w_l=\mathbf{0}$ 时 $\varphi\equiv 1$、$\alpha$ 均匀，AttnRes 起步为等权平均，经验验证训练平稳 [^src-moonshot-2026-attnres]。

开销：每 token $O(L^2 d)$ 算术、$O(Ld)$ 存储。普通训练下 $O(Ld)$ 与反向传播必存的激活完全重叠，零额外显存；启用激活重计算与流水线并行后，层输出必须常驻并跨 stage 传输，显存与通信都成为 $O(Ld)$[^src-moonshot-2026-attnres]。

### Block AttnRes

$L$ 层分 $N$ 块：块内保留标准残差（部分和），跨块只对块级表示与 embedding 做 softmax 注意力（式 5–6）。

### Stage B1 · 块内标准残差累加

$$\begin{aligned}
b_n &= \sum_{j\in B_n} f_j(h_j),\qquad b_n^i = b_n^{i-1} + f_l(h_l) && \text{// } [d]+[d]\to[d]\ \text{——块内沿用普通残差，部分和 } b_n^i \text{ 逐层演化，块末即块表示（式 5）}
\end{aligned}$$

### Stage B2 · 跨块注意力的源集合

$$\begin{aligned}
V &= [b_0;\,b_1;\,\ldots;\,b_{n-1}]^\top\ (i=1) && \text{// } [n,d]\ \text{——块首层只看已完成块；} b_0 = h_1 \text{ 保证 embedding 恒为源（式 6）}\\
V &= [b_0;\,\ldots;\,b_{n-1};\,b_n^{i-1}]^\top\ (i\ge 2) && \text{// } [n+1,d]\ \text{——块内后续层额外看当前部分和}
\end{aligned}$$

### Stage B3 · 同 Full 的 softmax 聚合

源数从 $l$ 降至 $n$ 或 $n+1$：$h_l=\sum_i \alpha_{i\to l}v_i$、$\varphi$ 与归一化不变；深度方向计算 $O(L^2)\to O(LN)$、存储 $O(Ld)\to O(Nd)$[^src-moonshot-2026-attnres]。

$N$ 在两个极端间插值：$N=L$ 退回 Full AttnRes，$N=1$ 退化为标准残差（embedding 孤立为 $b_0$）；$N\approx 8$ 即可恢复大部分收益[^src-moonshot-2026-attnres]。每层仅新增一个 RMSNorm 与一个 $w_l\in\mathbb{R}^d$，参数占比可忽略；48B 实验中模型其余组件全部不变，只改残差[^src-moonshot-2026-attnres]。

## 基础设施：规模下可用

**训练侧**：PP 下朴素全量传输的每 token 通信为 $\mathrm{Comm}_{\text{naïve}} = \tfrac{C(C-1)}{2}N_p d$（$C=PV$ 个 chunk，第 $j$ 个 chunk 已累积 $jN_p$ 块，式 7）；跨 stage 缓存后降至 $\tfrac{P(P-1)}{2}N_p d + (V-1)P^2 N_p d$（式 8），单次传输峰值 $O(C)\to O(P)$、$V\times$ 改善，可与稳态 1F1B 计算重叠[^src-moonshot-2026-attnres]。端到端训练开销：不开 PP 可忽略，开 PP 实测 <4%[^src-moonshot-2026-attnres]。

**推理侧**：伪查询与前向解耦，块内 $S$ 个查询可批处理。两阶段策略：Phase 1 批量读取缓存块表示 $[b_0;\ldots;b_{n-1}]$，算出块内全部 $S$ 层的跨块注意力并返回 softmax 统计量（max 与 LSE）；Phase 2 顺序计算块内部分和注意力，经 online softmax 合并，精确等价（Algorithm 1）[^src-moonshot-2026-attnres]。每 token 每层 I/O（Table 1，典型 $L{=}128$、$N{=}8$、$S{=}16$、$m{=}4$；不含层函数 $f_l$ 内部 I/O）：

| 残差机制 | 每 token 每层总 I/O |
|---|---|
| 标准残差 | $3d$ |
| mHC（$m=4$ 流） | $34d$ |
| Full AttnRes（两阶段摊销） | $(S+N)d = 24d$ |
| Block AttnRes（两阶段摊销） | $\left(\tfrac{N}{S}+5\right)d = 5.5d$ |

Block AttnRes 的残差流 I/O 显著低于 (m)HC 等残差流加宽方法，典型推理负载端到端延迟开销 <2%[^src-moonshot-2026-attnres]。长上下文 prefill 时，块表示缓存 $N\cdot T\cdot d$（128K 上下文、8 块为 15GB）沿序列维分片到 $P$ 个 TP 卡后为 $N\cdot(T/P)\cdot d\approx 1.9$GB/卡，叠加 16K chunked prefill 降至 <0.3GB/卡[^src-moonshot-2026-attnres]。Full AttnRes 两阶段摊销同理可得每层 $(S+N)d$（附录 B 式 11–17：跨块读 $dL(N-1)$ 摊到 $L$ 层得 $(N-1)d$，块内顺序读合计 $(S-1)d$）[^src-moonshot-2026-attnres]。

## 统一视图：与 Pre-LN/Post-LN、HC/mHC 的关系式

### 深度混合矩阵

所有残差变体均可写成 $h_l = \sum_{i=0}^{l-1} M_{i\to l}\,v_i$（$M\in\mathbb{R}^{L\times L}$ 为下三角），差异仅在权重来源（固定/静态学习/输入相关）与 $M$ 的 semiseparable 秩（§6.2）[^src-moonshot-2026-attnres]：

| 方法 | $M_{i\to l}$ | 权重 | 秩结构 |
|---|---|---|---|
| 标准残差 | $1$ | 固定 | 全 1 下三角，1-semiseparable |
| Highway | $g_{i+1}\gamma^{\times}_{i+1\to l}$，$\gamma^{\times}_{i\to l}=\prod_{j=i+1}^{l}(1-g_j)$ | 输入相关门控 | 1-semiseparable（stick-breaking） |
| (m)HC | $\beta_i^\top A^{\times}_{i+1\to l}\alpha_l$ | 动态 | $m$-semiseparable |
| Full AttnRes | $\alpha_{i\to l}$（softmax 注意力） | 动态 | 稠密，秩 $L$ |
| Block AttnRes | 同块源共享 $\alpha_{n\to l}$，块内每位置多一个部分和源 | 动态 | 介于 $N$ 与 $N+S$ |

### 与 HC/mHC 的关系式

(m)HC 单层更新（论文记号，流排成列，$H_l\in\mathbb{R}^{d\times m}$）：$H_l = H_{l-1}A_l + f_{l-1}(H_{l-1}\alpha_{l-1})\beta_{l-1}^\top$。展开递归得有效混合权重（式 10）：

$$M_{i\to l} = \beta_i^\top A^{\times}_{i+1\to l}\alpha_l,\qquad A^{\times}_{i\to j}:=\prod_{k=i+1}^{j}A_k$$

形状与逐元素含义：$\beta_i\in\mathbb{R}^m$ 是第 $i$ 层输出在 $m$ 条流上的分布（key，总结该层贡献）；$A^{\times}_{i+1\to l}\in\mathbb{R}^{m\times m}$ 是中间各层流间转移的累积乘积（深度相对位置算子）；$\alpha_l\in\mathbb{R}^m$ 是第 $l$ 层读流的查询；三者缩并出标量权重。$\alpha_l,\beta_l,A_l$ 与 [[hyper-connections]] 页的 $H_l^{pre},H_l^{post},H_l^{res}$ 一一对应（两处行列排列约定互为转置，该页流数记 $n$、本文记 $m$）[^src-moonshot-2026-attnres]。

论文由此给出统一视角：**(m)HC 是矩阵值状态的深度维线性注意力，AttnRes 是深度维 softmax 注意力**。$m$ 条流对应深度轴的状态扩展，mHC 的双随机约束（见 [[manifold-constrained-hyper-connections]]，经 [[sinkhorn-algorithm|Sinkhorn-Knopp]] 投影）用于稳定累积乘积 $A^{\times}$。文中引述第三方观察指出，将 $A^{\times}$ 换成单位矩阵仍有竞争力，说明流数 $m$ 的状态扩展本身贡献不小[^src-moonshot-2026-attnres]。若核可分解 $\varphi(q,k)=\phi(q)^\top\phi(k)$，深度注意力便塌缩回递归形式；这正是 MRLA–GLA 与 DDL–DeltaNet 对应的来源，也印证了 AttnRes 与 (m)HC 分属 softmax 与线性两支[^src-moonshot-2026-attnres]。

### Pre-LN/Post-LN 关系

Pre-LN 的等权累加使 $\|h_l\|$ 随深度 $O(L)$ 增长（$M$ 全 1 下三角的直接后果）；Post-LN 靠残差路径上重复归一化压住模长，但复合归一化使梯度随深度消失。AttnRes 用凸组合替换加性递归，同时避开 Pre-LN 的无界增长与 Post-LN 的重复缩放；其块边界重置累加，输出模长呈有界周期模式（Fig. 5b）[^src-moonshot-2026-attnres]。

## 理论性质：可证事实与作者论断分层

由定义可直接推出的：

1. 凸组合性：$\alpha_{i\to l}\ge 0$ 且 $\sum_i\alpha_{i\to l}=1$，故 $h_l$ 恒为前序源的凸组合，逐 token 向量满足 $\|h_l\|\le\max_i\|v_i\|$；
2. 零初始化退化：$w_l=\mathbf{0}\Rightarrow\alpha$ 均匀 $1/l$，起步为等权平均；
3. 插值退化：$N=L$ 即 Full，$N=1$ 即标准残差，$S=1$（Fig. 6 记号）等价 Full；
4. 秩阶梯：标准残差/Highway 1-semiseparable → (m)HC $m$ → Block $\in[N,\,N+S]$ → Full $L$，均由递归展开直接可得；
5. 核分解塌缩：$\varphi(q,k)=\phi(q)^\top\phi(k)$ 时深度注意力可写成递归（线性注意力形式）。

作者论断/经验结果（论文未给定理证明）：

Pre-LN 稀释得到缓解，48B 训练中输出模长有界呈周期模式、梯度分布更均匀（Fig. 5）；某些源（尤其 embedding）跨输入持续获得高权重，类似序列注意力的 attention sink（Fig. 8）；静态系数的 DenseFormer 与基线打平，表明输入相关权重是增益来源（Table 4）。

## 实验证据

### Scaling laws（Table 2、Fig. 4）

五个尺寸各训练基线、Block（$N\approx 8$）、Full 与 mHC(-lite) 四个变体；超参按基线选定（对基线有利的保守比较），上下文 8192，激活参数不含 embedding：

| 激活参数 | tokens | 基线 | Block | Full | mHC(-lite) |
|---|---|---|---|---|---|
| 194M | 38.7B | 1.931 | 1.909 | 1.899 | 1.906 |
| 241M | 45.4B | 1.895 | 1.875 | 1.874 | 1.869 |
| 296M | 62.1B | 1.829 | 1.809 | 1.804 | 1.807 |
| 436M | 87.9B | 1.766 | 1.746 | 1.737 | 1.747 |
| 528M | 119.0B | 1.719 | 1.693 | 1.692 | 1.694 |

拟合曲线 $L=A\times C^{-\alpha}$（$C$ 单位 PFLOP/s-days）：基线 $1.891\times C^{-0.057}$，Block $1.870\times C^{-0.058}$，Full $1.865\times C^{-0.057}$；三者斜率几乎一致，AttnRes 全程更低[^src-moonshot-2026-attnres]。按拟合曲线，5.6 PFLOP/s-days 处 Block 1.692 对基线 1.714，作者折算为等效 1.25× 计算优势；Full 与 Block 差距随规模收窄，最大尺寸仅差 0.001（Table 2 第 5 行：1.692 vs 1.693）[^src-moonshot-2026-attnres]。对 mHC(-lite)：正文称 Full AttnRes outperforms mHC，Block 以更低 I/O（5.5d 对 34d）与之持平[^src-moonshot-2026-attnres]；按 Table 2 逐档数字，Full 在 5 档中 4 档更低（仅 241M 档例外，1.874 对 1.869），Block 在 436M/528M 档更低、194M–296M 档略高 0.002–0.006，正文概括与表格数字存在出入[^src-moonshot-2026-attnres]。

### 48B 主结果（Table 3）

架构采用 Kimi Linear（3:1 KDA/MLA 交错 + MoE，27 个 Transformer 块即 54 层，8/256 路由专家 + 1 共享专家，48B 总/3B 激活）；Block AttnRes 每块 6 层，含 9 块及 embedding 共 10 个深度源；数据与配方沿用 Kimi Linear 1.4T run——上下文 4096、Muon、WSD、全局 batch 8M token，经 1T 预训练与约 400B mid-training，随后扩展至 32K 上下文（MLA 为 NoPE，无需 YaRN 类修改）[^src-moonshot-2026-attnres]。

同配方对照下，15 项基准 14 升 1 平：MMLU 73.5→74.6、MMLU-Pro 52.2→52.2（持平）、GPQA-Diamond 36.9→44.4（+7.5）、BBH 76.3→78.0、ARC-Challenge 64.6→65.7、HellaSwag 83.2→83.4、TriviaQA 69.9→71.8、GSM8K 81.7→82.4、MGSM 64.9→66.1、MATH 53.5→57.1（+3.6）、CMath 84.7→85.1、HumanEval 59.1→62.2（+3.1）、MBPP 72.0→73.9、CMMLU 82.0→82.9、C-Eval 79.6→82.5[^src-moonshot-2026-attnres]。多步推理与代码任务增幅更大，深度信息流改善利于组合任务[^src-moonshot-2026-attnres]。

### 训练动力学（Fig. 5）

1T token 训练中验证损失全程更低，decay 段差距拉大。基线各块输出模长因 Pre-LN 稀释随深度单调增长，Block AttnRes 呈块内有界周期模式；基线最早层梯度显著偏大，Block AttnRes 的 softmax 竞争使梯度分布明显更均匀[^src-moonshot-2026-attnres]。

### 消融（Table 4，436M/87.9B 档，$L_b{=}16$、$H{=}16$）

| 变体 | 损失 |
|---|---|
| 基线（PreNorm） | 1.766 |
| DenseFormer（静态系数跨层访问） | 1.767 |
| mHC | 1.747 |
| Full AttnRes | 1.737 |
| Full + 输入相关查询 | 1.731 |
| Full − 查询/键（静态标量混合） | 1.749 |
| Full，sigmoid 替代 softmax | 1.741 |
| Full − 键上 RMSNorm | 1.743 |
| 滑窗 SWA（$W=1+8$） | 1.764 |
| Block（$S=4$，$N=8$） | 1.746 |
| Block + 多头深度聚合（$H{=}16$） | 1.752 |
| Block − 键上 RMSNorm | 1.750 |

静态权重的 DenseFormer 与基线持平（1.767 对 1.766），输入相关才是增益来源；显式内容相关选择更进一步，Full 1.737 低于 mHC 1.747，Block 1.746 与 mHC 持平且每层 I/O 更低；滑窗只看最近 8 层与 embedding 仅得 1.764，选择性访问远层比多看近邻重要；多头深度聚合反而变差（1.752 对 1.746），通道间最优深度混合基本一致，一层输出相关时整体相关[^src-moonshot-2026-attnres]。块大小扫描（Fig. 6，同模型）：基线 1.766，$S{=}1$（Full）1.737，$S{=}2/4/8$ 落在约 1.746，$S{=}16/32$ 向基线回落；生产配置据此固定 $N\approx 8$[^src-moonshot-2026-attnres]。

### 最优架构扫描（Fig. 7）

固定算力 $\approx 6.5\times 10^{19}$ FLOPs、激活参数 $\approx 2.3\times 10^8$、$d_{ff}/d_{model}\approx 0.45$，遍历 $d_{model}/L_b\in\{15,30,45,60,75\}\times H/L_b\in\{0.3,\ldots,0.7\}$ 共 25 种配置：AttnRes 在全部 25 格均低于基线，幅度 0.019–0.063；两者最优均在 $H/L_b\approx 0.3$，但基线最优在 $d_{model}/L_b\approx 60$（损失 1.847），AttnRes 移至 $\approx 45$（1.802）——同等预算下更窄更深[^src-moonshot-2026-attnres]。该深度偏好受推理延迟牵制，作为架构选择的诊断输入[^src-moonshot-2026-attnres]。

### 学到的权重模式（Fig. 8）

在 16 头模型（$L_b{=}16$，16 attention + 16 MLP 层；Block 变体 $N{=}8$）中，权重呈对角占优，局部性仍是主信息通路；embedding 源持续保有非平凡权重（深度注意力汇）；pre-attention 层感受野更宽，pre-MLP 层更依赖近邻；Block 变体保留全部结构且权重更尖锐，块压缩起隐式正则作用[^src-moonshot-2026-attnres]。

## 边界（论文自述）

Full AttnRes 的 $O(Ld)$ 跨层聚合显存在当前硬件下对大模型不可行，因而引入 Block 变体，更细块粒度留待显存限制放宽[^src-moonshot-2026-attnres]；训练瓶颈主要在 PP 跨 stage 通信而非本地带宽，互连改进后 Full 路线可行[^src-moonshot-2026-attnres]。深度维 $O(L^2)$ 注意力以较小深度（$L<1000$）为前提，线性复杂度的深度注意力是未来方向[^src-moonshot-2026-attnres]。输入相关查询损失更低（1.731），但因需要 $d\times d$ 投影且解码时引入顺序访存，故默认采用学习型静态查询[^src-moonshot-2026-attnres]。48B 对照仅含基线与 Block AttnRes 两变体，Full 与 mHC(-lite) 的对照在 $\leq$528M 的 scaling law 档完成[^src-moonshot-2026-attnres]。

## 相关页面

[[hyper-connections]] 为多流残差递归，被本文统一为深度维线性注意力；[[manifold-constrained-hyper-connections]] 是 HC 的双随机稳定版，scaling law 对照所用 mHC(-lite)（arXiv:2601.05732）为其免 Sinkhorn 变体，与 mHC 原文（arXiv:2512.24880）分属两篇，Table 2 用前者、Table 4 用后者；[[source-attention-is-all-you-need-1706-03762]] 是序列维度 softmax 注意力的出处与时间–深度对偶的类比来源。

[^src-moonshot-2026-attnres]: [[source-moonshot-2026-attnres]]
