---
title: "Agent Attention（(Q, A, K, V) 四元组注意力）"
type: technique
tags:
  - attention
  - efficient-attention
  - linear-attention
  - vision
  - transformer
created: 2026-10-24
last_updated: 2026-10-24
source_count: 2
confidence: medium
status: active
---

# Agent Attention

Agent Attention（Han et al., ECCV 2024, arXiv:2312.08874）把三元组 $(Q,K,V)$ 扩成四元组 $(Q,A,K,V)$，加入 $n$ 个 agent token $A$：$A$ 先以 query 身份从 $K,V$ 聚合出 agent feature，再以 key 身份广播回每个 query。两段均为标准 softmax 注意力，$Q$ 与 $K$ 之间不再两两配对（式 3，Fig. 3a）[^src-2312-08874-agent-attention]。

## 问题与动机

Softmax 注意力逐 query-key 对算相似度，复杂度 $\mathcal{O}(N^2)$，在全局感受野的视觉任务上计算量不可接受。既有做法分两类：Swin 限制在局部窗口，PVT 在 $K,V$ 上做稀疏，代价均为牺牲长程建模（§2）[^src-2312-08874-agent-attention]。线性注意力换掉 softmax，用 $\phi$ 分别映射 $Q$ 与 $K$，按结合律改写计算顺序把复杂度降到 $\mathcal{O}(N)$；但 ReLU 一类简单 $\phi$ 掉点明显，复杂 $\phi$（Performer、Nyströmformer、SOFT）引入额外开销（§3.2）[^src-2312-08874-agent-attention]。

Agent attention 不削减感受野、不近似 softmax，而是利用注意力权重的冗余：Fig. 1a 中语义相同的一批 query（如天空区域）所需信息几乎一致，可由少数 agent token 代收代发[^src-2312-08874-agent-attention]。

## 两段 softmax 与广义线性注意力

记 $\sigma(\cdot)$ 为 softmax，$\phi$ 为线性注意力的映射函数：

$$O^{\rm S}=\sigma(QK^{\top})V \triangleq \operatorname{Attn}^{\rm S}(Q,K,V),\qquad O^{\phi}=\phi(Q)\phi(K)^{\top}V \triangleq \operatorname{Attn}^{\phi}(Q,K,V)$$

Agent attention 写作两段 softmax 的嵌套，再展开成一次核化线性注意力（式 3–4）[^src-2312-08874-agent-attention]：

$$\begin{aligned}
O^{\rm A} &= \operatorname{Attn}^{\rm S}\!\Big(Q,\;A,\;\underbrace{\operatorname{Attn}^{\rm S}(A,K,V)}_{\text{Agent Aggregation}}\Big) && \text{// 段 1 造 agent feature，段 2 广播}\\
&= \sigma(QA^{\top})\,\sigma(AK^{\top})\,V \;=\; \phi_q(Q)\,\phi_k(K)^{\top}V && \text{// 令 } \phi_q(Q)=\sigma(QA^{\top}),\ \phi_k(K)=\sigma(AK^{\top})^{\top}
\end{aligned}$$

式 4 对任意 $n$ 成立，$n$ 不改变等价关系，只改变 $\phi$ 的参数化与计算量。softmax 与 linear attention 因此不是两个可切换的算子，而是同一算子的两种读法：外部看是两段 softmax（式 3），代数上是以 $A$ 参数化的核函数线性注意力（式 4）[^src-2312-08874-agent-attention]。

两种读法的计算顺序重合：线性注意力按结合律先算 $\phi_k(K)^{\top}V$（$[n,N]\times[N,C]\to[n,C]$）再左乘 $\phi_q(Q)$，与式 3 的 aggregate → broadcast 同序，$n$ 即该核映射的维数。复杂度因此是 $\mathcal{O}(Nnd)$ 而非 $\mathcal{O}(N^2)$，节省来自中间量只有 $n$ 个而不是 $N$ 个[^src-2312-08874-agent-attention]。

[INFERENCE] $\phi$ 是否依赖输入集合，由 agent token 的取法决定。取可学习参数（Table 17 的 Static 行，Acc. 82.2）时 $A$ 与输入无关，式 4 退化为固定核的线性注意力；取池化（Dynamic，82.6）时 $\phi_q$ 依赖整层 $Q$，超出「逐 token 映射」的线性注意力设定。这一步是代换的结果，论文未作此区分。

## 数据流与维度

**维度符号**：$N$ 序列长度（token 数）；$C$ 模块通道维；$d$ 单头通道维；$n$ agent token 数；$k=3$ 为 DWC 的卷积核边长。形状按论文记法写作 $[\text{行}, \text{列}]$，$Q,K,V\in[N,C]$，$A\in[n,C]$，$B_1\in[n,N]$，$B_2\in[N,n]$（式 6）。论文按单头书写，$/\sqrt{d}$ 取自式 1 的相似度定义，多头在 batch 维展开为 [INFERENCE]。

### Stage 1 · 池化取 agent token

$$\begin{aligned}
A &= \operatorname{Pooling}(Q) && \text{// } [N, C] \xrightarrow{\text{沿 } N \text{ 维池化}} [n, C]\ \text{——} N \text{ 个 query 压成 } n \text{ 个 agent token}
\end{aligned}$$

池化窗口按 stage 调整：ImageNet 上四个 stage 取 $[9,16,49,49]$，密集预测任务上 Agent-PVT 取 $[144,256,784,784]$、Agent-Swin 取 $[81,144,196,49]$（Appendix D.2）[^src-2312-08874-agent-attention]。论文未给出池化核的具体实现形式。

### Stage 2 · Agent Aggregation：$A$ 作 query

$$\begin{aligned}
S_1 &= \sigma\!\left(AK^{\top}/\sqrt{d}+B_1\right) && \text{// } [n, \bcancel{C}]\times[\bcancel{C}, N]\to[n, N]\ \text{——消去 } C\text{，每行是一个 agent token 对全部 } N \text{ 个 key 的打分}\\
V^{A} &= S_1 V && \text{// } [n, \bcancel{N}]\times[\bcancel{N}, C]\to[n, C]\ \text{——沿 } N \text{ 加权求和，} N \text{ 个 value 收成 } n \text{ 个 agent feature}
\end{aligned}$$

### Stage 3 · Agent Broadcast：$A$ 作 key

$$\begin{aligned}
S_2 &= \sigma\!\left(QA^{\top}/\sqrt{d}+B_2\right) && \text{// } [N, \bcancel{C}]\times[\bcancel{C}, n]\to[N, n]\ \text{——同一个 } A \text{ 改作 key，每个 query 得到 } n \text{ 个权重}\\
O^{A} &= S_2 V^{A} && \text{// } [N, \bcancel{n}]\times[\bcancel{n}, C]\to[N, C]\ \text{——每个 query 只读 } n \text{ 个 agent feature}
\end{aligned}$$

两段共用同一组 $A$：第一段作 query、第二段作 key，$V^{A}$ 只有一份，被全部 $N$ 个 query 读取（Fig. 3a）[^src-2312-08874-agent-attention]。

### Stage 4 · 汇合：补特征多样性

$$\begin{aligned}
O &= O^{A} + \operatorname{DWC}(V) && \text{// } [N, C]+[N, C]\to[N, C]\ \text{——逐通道深度卷积，按位置混合相邻 token}
\end{aligned}$$

DWC 沿用 Flatten Transformer 的 focused linear attention 做法，补回线性注意力丢失的特征多样性（§4.2）[^src-2312-08874-agent-attention]。Agent bias 与 DWC 一并写进模块式（式 6）：$O=\sigma(QA^{\top}+B_2)\,\sigma(AK^{\top}+B_1)\,V+\operatorname{DWC}(V)$。

### Agent bias 的参数化

$B_1,B_2$ 不作 $[n,N]$ 与 $[N,n]$ 的可学习张量，各由三个分量构成：列偏置 $B_{1c}\in\mathbb{R}^{n\times 1\times w}$、行偏置 $B_{1r}\in\mathbb{R}^{n\times h\times 1}$、块偏置 $B_{1b}\in\mathbb{R}^{n\times h_0\times w_0}$，计算时各自 resize 到 $[n,h,w]$ 后相加、reshape 成 $[n,N]$（Appendix B）[^src-2312-08874-agent-attention]。$h_0,w_0$ 远小于特征图高宽 $h,w$。

## 复杂度

单 query 路径只经过 $n$ 个 agent token，模块总开销为（式 7）[^src-2312-08874-agent-attention]：

$$\Omega=\underbrace{4NC^2}_{\rm Proj}+\underbrace{NC}_{\rm Get\ agents}+\underbrace{2nNC+2NnC}_{\rm Agent\ Attention}+\underbrace{k^2NC}_{\rm DWC}$$

$N$ 只以一次幂出现，注意力部分为 $\mathcal{O}(Nnd)$；两段 softmax 的矩阵乘法各为 $2nNC$ 与 $2NnC$，合计 $4NnC$。相对 softmax 注意力的 $\mathcal{O}(N^2)$，收益条件是 $n\ll N$。式中不含 softmax 归一化、池化与 gather 的访存成本，也未给空间复杂度；效率证据以 FLOPs 与实测 wall-clock 给出（Fig. 5、Fig. 7）[^src-2312-08874-agent-attention]。

## 证据

### ImageNet-1K 分类（Table 13，Table 1 为其子集）

替换 Softmax 注意力，同配方从头训练 300 epoch（Appendix D.1）。Table 1 与 Table 13 的同名行数字一致[^src-2312-08874-agent-attention]：

| 模型 | Reso | #Params | FLOPs | Top-1 |
|---|---|---|---|---|
| DeiT-T → Agent-DeiT-T | $224^2$ | 5.7M → 6.0M | 1.2G → 1.2G | 72.2 → 74.9 (+2.7) |
| DeiT-S → Agent-DeiT-S | $224^2$ | 22.1M → 22.7M | 4.6G → 4.4G | 79.8 → 80.5 (+0.7) |
| DeiT-B → Agent-DeiT-B | $224^2$ | 86.6M → 87.2M | 17.6G → 17.6G | 81.8 → 82.0 (+0.2) |
| PVT-T → Agent-PVT-T | $224^2$ | 13.2M → 11.6M | 1.9G → 2.0G | 75.1 → 78.4 (+3.3) |
| PVT-S → Agent-PVT-S | $224^2$ | 24.5M → 20.6M | 3.8G → 4.0G | 79.8 → 82.2 (+2.4) |
| PVT-M → Agent-PVT-M | $224^2$ | 44.2M → 35.9M | 6.7G → 7.0G | 81.2 → 83.4 (+2.2) |
| PVT-L → Agent-PVT-L | $224^2$ | 61.4M → 48.7M | 9.8G → 10.4G | 81.7 → 83.7 (+2.0) |
| Swin-T → Agent-Swin-T | $224^2$ | 29M | 4.5G | 81.3 → 82.6 (+1.3) |
| Swin-S → Agent-Swin-S | $224^2$ | 50M | 8.7G | 83.0 → 83.7 (+0.7) |
| Swin-B → Agent-Swin-B | $224^2$ | 88M | 15.4G | 83.5 → 84.0 (+0.5) |
| CSwin-T → Agent-CSwin-T | $224^2$ | 23M → 21M | 4.3G → 4.3G | 82.7 → 83.1 (+0.4) |
| CSwin-S → Agent-CSwin-S | $224^2$ | 35M → 33M | 6.9G → 6.8G | 83.6 → 83.9 (+0.3) |
| CSwin-B → Agent-CSwin-B | $224^2$ | 78M → 73M | 15.0G → 14.9G | 84.2 → 84.7 (+0.5) |
| CSwin-B↑$384^2$ → Agent-CSwin-B↑$384^2$ | $384^2$ | 78M → 73M | 47.0G → 46.3G | 85.4 → 85.8 (+0.4) |

§5.1 正文另称 Agent-PVT-S 以 30% 的参数量、40% 的 FLOPs 超过 PVT-L（Top-1 82.2 vs 81.7）。按 Table 1 复算：参数量 20.6M / 61.4M = 33.6%，FLOPs 4.0G / 9.8G = 40.8%，FLOPs 一项与正文一致，参数量一项正文低于表值[^src-2312-08874-agent-attention]。

### 检测与分割

COCO 检测（Table 2，Mask R-CNN 与 Cascade Mask R-CNN，FLOPs 按 $1280\times800$ 计）：Agent-PVT-T box AP 41.4 vs PVT-T 36.7，Agent-PVT-L 46.9 vs PVT-L 42.9，Agent-Swin-S 47.2 vs Swin-S 45.7；§5.2 归纳 box AP 增幅为 +3.9 至 +4.7（PVT 系）与最高 +1.5（Swin 系）[^src-2312-08874-agent-attention]。RetinaNet 与完整 COCO 结果在 Table 14、Table 15。

ADE20K 分割（Table 3，FLOPs 按 $512\times2048$ 计）：SemanticFPN 下 Agent-PVT-T mIoU 40.18 vs PVT-T 36.57（+3.61），UperNet 下 Agent-Swin-T 46.68 vs Swin-T 44.51（+2.17）[^src-2312-08874-agent-attention]。完整表在 Table 16。

### 推理速度与高分辨率

$224^2$ 输入下的实测加速：i5 CPU 上 1.7–2.1 倍，RTX3090 与 A100 上 1.4–1.7 倍，同时 Top-1 更高（Fig. 5）[^src-2312-08874-agent-attention]。与其他线性注意力方法的同机对比见 Fig. 9，源文只给曲线，无数值表。

$1024^2$ 输入下 Agent-DeiT 的 FLOPs 比 DeiT 少 75%（Fig. 7b）[^src-2312-08874-agent-attention]。从零训练的分辨率扩展（Table 5）：Agent-DeiT-B $224^2$ 82.0 vs DeiT-B 81.8；Agent-DeiT-S 在 $448^2$、17.7G FLOPs 下取 83.1，DeiT-S 在 $416^2$、18.8G 下取 82.9，前者参数量 23.1M、后者 22.2M；Agent-PVT-M $256^2$ 83.8 (+2.1) vs PVT-M $256^2$ 82.2。Fig. 7c 另给出 $256^2$–$384^2$ 微调 30 epoch 的曲线。Table 13 中 Agent-Swin-S 在 $288^2$ 取 84.1、Agent-Swin-B 在 $384^2$ 取 84.9（同分辨率 Swin-B 为 84.5）。

### 消融

以 Agent-Swin-T 为基准，Swin-T 为 81.3；Table 6 的 Diff. 列以 Ours（82.6）为参考[^src-2312-08874-agent-attention]：

| 配置 | FLOPs | Acc. | Diff.（原文，相对 82.6） |
|---|---|---|---|
| Vanilla Linear Attention | 4.5G | 77.8 | −4.8 |
| + Agent Attention | 4.5G | 79.0 | −3.6 |
| + Agent Bias | 4.5G | 81.1 | −1.5 |
| + DWC | 4.5G | 82.6 | — |

同表另给出 Swin-T w/o PE 80.1（−2.5）、+RPE 81.3（−1.3）、+DWC 81.6（−1.0）。agent bias 承担 Swin 中 RPE 的位置：softmax 换成纯线性注意力先掉到 77.8（−3.5 于 Swin-T），换 agent attention 到 79.0，加 bias 到 81.1，加 DWC 到 82.6；DWC 单独加到 Swin-T 只从 81.3 到 81.6（+0.3，§5.6）[^src-2312-08874-agent-attention]。

窗口大小（Table 4）：Agent-Swin-T 的窗口从 $7^2$ 放大到 $56^2$，Acc. 依次 82.0 / 82.2 / 82.4 / 82.6，FLOPs 恒为 4.5G；Swin-T 在 $7^2$ 为 81.3[^src-2312-08874-agent-attention]。

agent token 数（Table 7，四个 stage 依次给出）：$[49,49,49,49]$ 为 4.7G / 82.6，$[9,16,49,49]$ 为 4.5G / 82.6（默认），$[9,16,25,49]$ 为 4.5G / 82.2，$[4,9,49,49]$ 为 4.5G / 82.4。浅层减 agent token 不掉点、深层减会掉点，与 CSwin 的 stripe width 取法一致（§5.6）[^src-2312-08874-agent-attention]。

agent token 取法（Table 17）：可学习参数（Static）82.2，池化 82.6，DWC 82.6，DAT 的 deformed points 82.7，token merging 82.6（4.6G）[^src-2312-08874-agent-attention]。

施加位置（Table 18）：只换 stage 1 为 81.7，stage 1–2 为 81.8，stage 1–3 为 82.6，四个 stage 全换为 82.5[^src-2312-08874-agent-attention]。

与其他线性注意力对比（Table 8）：DeiT-T 上 Hydra 68.3、Efficient Attention 70.2、Castling-ViT 的 linear angular 70.8、FLatten 的 focused linear 74.1（6.1M / 1.1G）、agent attention 74.9（6.0M / 1.2G）；Swin-T 上依次 80.7 / 81.0 / 79.4 / 82.1 / 82.6[^src-2312-08874-agent-attention]。

### Stable Diffusion

AgentSD 把 ToMeSD 中的 softmax 注意力换成 agent attention，agent token 由 token merging 产生，无需额外训练；无训练时不能用 agent bias 与 DWC，改用 $O=\sigma(QA^{\top})\sigma(AK^{\top})V+kV$（$k$ 为预设常数），并把第二段 softmax 的缩放从 $d^{-0.5}$ 调大（式 10，Appendix C.1）[^src-2312-08874-agent-attention]。定量结果（Table 9，SD v1.5，2000 张 ImageNet-1K 类图，50 步 PLMS，cfg 7.5，单卡 RTX4090）：

| 方法 | $r$ | FID | s/img | GB/img |
|---|---|---|---|---|
| SD | 0 | 28.84 | 2.62 | 3.13 |
| ToMeSD | 0.1 | 28.64 | 2.40 | 2.55 |
| ToMeSD | 0.4 | 28.74 | 1.71 | 1.69 |
| ToMeSD | 0.5 | 29.01 | 1.53 | 1.47 |
| AgentSD | 0.1 | 27.79 | 1.97 | 1.77 |
| AgentSD | 0.2 | 27.77 | 1.80 | 1.60 |
| AgentSD | 0.4 | 28.15 | 1.54 | 1.55 |
| AgentSD | 0.5 | 28.42 | 1.42 | 1.21 |

正文的 1.84 倍与 1.69 倍分别对应 $2.62/1.42$（SD 对 AgentSD $r=0.5$）与 $2.40/1.42$（ToMeSD $r=0.1$ 对 AgentSD $r=0.5$）；「同速下 FID 低 0.9」只能由同合并比 $r=0.2$ 的两行复现（28.68 → 27.77，差 0.91），两行速度 2.15 与 1.80 s/img 并不相等（§5.4，Table 9，Fig. 6a）[^src-2312-08874-agent-attention]。$k$ 的消融（Table 10）：$k=0$ 28.80、0.025 28.67、0.075 28.42、0.15 28.61；第二段 softmax 缩放（Table 11）：$d^{-0.5}$ 28.86、$d^{-0.25}$ 28.64、$d^{-0.15}$ 28.42、$d^{-0.05}$ 28.60；施加步数（Table 12）：前 20% 为 28.58 / 1.50 s、前 40% 为 28.42 / 1.42 s、前 60% 为 28.83 / 1.39 s、前 80% 为 29.77 / 1.34 s。微调场景（Dreambooth）下可全程施加，提速 2.2 倍，微调时间与显存各降约 15%，RTX4090 上 12GB 以内约 7 分钟（Appendix C.4）。

## 与 MiTA 共享专家的关系

MiTA 的共享专家由两步组成（Algorithm 1 第 2、4、9、11 行）：$\tilde Q=\operatorname{AdaptiveAvgPool}(Q,\text{output\_size}=m)$、$\tilde V=V\operatorname{softmax}(K^{\top}\tilde Q/\sqrt d)$，再取 $O^{\rm share}=\operatorname{FlashAttention}(Q,\tilde Q,\tilde V)$[^src-2602-01219-mita]。这两步与 Agent attention 的 Stage 2（agent aggregation）、Stage 3（agent broadcast）逐项同型：同一个 $A$ 先作 query 从全部 $K,V$ 收信息，再作 key 被全部 query 读取。差别在 $A$ 的取法与后续：Agent attention 取 $\operatorname{Pooling}(Q)$，广播后即结束（Table 17 把池化记为 Dynamic）；MiTA 取分段平均池化，并在共享专家之外再加 top-$k$ 形变专家与 $N$-to-$m$ 路由[^src-2312-08874-agent-attention][^src-2602-01219-mita]。

MiTA 的分类表把 Agent attention 记为 Compression / 1 个共享 MLP / landmark testing / all-to-one；一手源对得上：agent token 直接取自 $Q$ 的池化、不训练，两段之外没有路由[^src-2312-08874-agent-attention][^src-2602-01219-mita]。上段「同源」判断是本 wiki 的对照分析，两篇源文均未互引为同一机制的来源。

## 边界

- 实验限于视觉任务（ImageNet-1K、COCO、ADE20K、Stable Diffusion），无语言建模与长序列基准；$\mathcal{O}(Nnd)$ 相对 $\mathcal{O}(N^2)$ 的收益只在视觉分辨率（$224^2$–$1024^2$）下用 FLOPs 与 wall-clock 体现，未给出与 $N$ 成比例的序列长度扫描[^src-2312-08874-agent-attention]。
- agent token 数只在 Agent-Swin-T 上做消融（Table 7），其他骨干沿用同一组 $[9,16,49,49]$，未给跨骨干的最优 $n$。
- 与 GPViT、GRL 的区别只有定性论证（Appendix A），无实验对照；源文对三者的差异归结为 agent token 是否取自 query space（$A=f(Q)$）以及能否免训练迁移[^src-2312-08874-agent-attention]。
- COCO 训练用 80k 子集、35k 验证，不是完整的 118K/5K 划分（Appendix D.2）；Table 2 的表注按 $1280\times800$ 计 FLOPs，正文称 mAP 以 $(3,1333,800)$ 测试。
- 式 7 的复杂度只到 FLOPs 量级，softmax 归一化、池化、gather 的访存与 kernel 实现未计入，实测速度只有曲线（Fig. 5、Fig. 9）没有表格数值[^src-2312-08874-agent-attention]。

## 相关页面

- [[source-2312-08874-agent-attention]] — 源摘要
- [[mita-attention]] — 共享专家与 landmark query 的后续机制
- [[fast-weight-scaling-taxonomy]] — 本方法在五维分类中的坐标
- [[cbsa]] · [[representative-token-extraction]] — 代表性 token 取法的另两条实例链
- [[linear-attention-unified-framework]] — 核函数级线性注意力统一
- [[adaptive-graph-agent-attention]] — 另一类 agent token 中介注意力（时空图）

[^src-2312-08874-agent-attention]: [[source-2312-08874-agent-attention]]
[^src-2602-01219-mita]: [[source-2602-01219-mita]]
