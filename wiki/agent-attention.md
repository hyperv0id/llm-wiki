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
last_updated: 2026-11-22
source_count: 2
confidence: medium
status: active
---

# Agent Attention

Agent Attention（Han et al., ECCV 2024, arXiv:2312.08874）把三元组 $(Q,K,V)$ 扩成四元组 $(Q,A,K,V)$，加入 $n$ 个 agent token $A$：$A$ 先以 query 身份从 $K,V$ 聚合出 agent feature，再以 key 身份广播回每个 query。两段均为标准 softmax 注意力，$Q$ 与 $K$ 之间不再两两配对（式 3，Fig. 3a）[^src-2312-08874-agent-attention]。

## 问题与动机

Softmax 注意力逐 query-key 对算相似度，复杂度 $\mathcal{O}(N^2)$，在全局感受野的视觉任务上计算量不可接受。既有做法分两类：Swin 限制在局部窗口，PVT 在 $K,V$ 上做稀疏，代价均为牺牲长程建模（§2）[^src-2312-08874-agent-attention]。线性注意力换掉 softmax，用 $\phi$ 分别映射 $Q$ 与 $K$，按结合律改写计算顺序把复杂度降到 $\mathcal{O}(N)$；但 ReLU 一类简单 $\phi$ 掉点明显，复杂 $\phi$（Performer、Nyströmformer、SOFT）引入额外开销（§3.2）[^src-2312-08874-agent-attention]。

Agent attention 不削减感受野、不近似 softmax，而是利用注意力权重的冗余：同一语义区域（如天空）里的一批 query 所需信息几乎一致，可由少数 agent token 代收代发（Fig. 1a）[^src-2312-08874-agent-attention]。

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

[INFERENCE] $\phi$ 是否依赖输入集合，由 agent token 的取法决定。取可学习参数时 $A$ 与输入无关，式 4 退化为固定核的线性注意力；取池化时 $\phi_q$ 依赖整层 $Q$，超出「逐 token 映射」的线性注意力设定（Table 17 的 Static 行 82.2、Dynamic 行 82.6）。这一步是代换的结果，论文未作此区分。

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

**设置**：ImageNet-1K 从头训练 300 epoch，只把 softmax 注意力换成 agent attention，其余配方不变；检测、分割、扩散分别用 COCO、ADE20K 与 SD v1.5 [^src-2312-08874-agent-attention]。

**定位与主结论**：同等算力下是真精度提升，不是低开销逼近。DeiT、PVT、Swin、CSwin 四族 14 组配置的 Top-1 全部更高，涨幅 +0.2（DeiT-B）至 +3.3（PVT-T）；PVT 系同时在缩参数（PVT-S 24.5M → 20.6M，PVT-L 61.4M → 48.7M），COCO box AP 上 PVT 系 +3.9 至 +4.7、Swin 系最高 +1.5，ADE20K mIoU +3.61（SemanticFPN/PVT-T）与 +2.17（UperNet/Swin-T）。同为线性注意力时优势同样成立：DeiT-T 上 Hydra 68.3、FLatten 74.1、agent attention 74.9。开销一侧，$224^2$ 实测 1.4–2.1 倍加速且 Top-1 同时更高，$1024^2$ 时 FLOPs 比 DeiT 少 75%；分辨率扩展在近似等 FLOPs 下仍占优（Agent-DeiT-S $448^2$、17.7G 取 83.1，DeiT-S $416^2$、18.8G 取 82.9）[^src-2312-08874-agent-attention]。

| 骨干 | #Params | FLOPs | Top-1 |
|---|---|---|---|
| DeiT-T → Agent-DeiT-T | 5.7M → 6.0M | 1.2G → 1.2G | 72.2 → 74.9 |
| PVT-S → Agent-PVT-S | 24.5M → 20.6M | 3.8G → 4.0G | 79.8 → 82.2 |
| CSwin-B↑$384^2$ → Agent-CSwin-B↑$384^2$ | 78M → 73M | 47.0G → 46.3G | 85.4 → 85.8 |

**涨点归因**：主故事的两段 softmax 本身只把纯线性注意力从 77.8 拉回 79.0，相对完整模块 82.6 还差 3.6 点；这部分由 agent bias（81.1）与 DWC（82.6）补上。DWC 单独加到 Swin-T 只 +0.3，说明它只能在线性注意力替换后起作用。施加位置到 stage 1–3 即饱和（82.6，四个 stage 全换 82.5）；agent token 数浅层可减、深层不能减（$[9,16,49,49]$ 与四层全 49 同为 82.6，深层减到 25 掉到 82.2）；取法上池化、token merging、DAT 形变点差别在 0.1 点内，可学习参数退到 82.2 [^src-2312-08874-agent-attention]。

**证据缺口与负结果**：全部实验限于视觉任务，没有语言建模与长序列基准，$\mathcal{O}(Nnd)$ 的优势只在 $224^2$–$1024^2$ 分辨率下用 FLOPs 与 wall-clock 体现；agent token 数只在 Agent-Swin-T 上消融，其他骨干沿用同一组取法；与 GPViT、GRL 只有定性论证，没有对照。正文称 Agent-PVT-S 用 30% 参数量超过 PVT-L，按 Table 1 合计口径是 20.6M/61.4M = 33.6%（FLOPs 4.0G/9.8G = 40.8% 与正文一致，§5.1），两数并列。免训练迁移到 Stable Diffusion 时同合并比下 FID 27.77 对 ToMeSD 28.68、速度 1.80 对 2.15 s/img，但正文的「1.84 倍 / 1.69 倍」与「同速下 FID 低 0.9」分别只能由不同行复现，两行速度并不相等；施加步数超过 60% 后 FID 退化到 29.77 [^src-2312-08874-agent-attention]。

## 与 MiTA 共享专家的关系

MiTA 的共享专家由两步组成：$\tilde Q=\operatorname{AdaptiveAvgPool}(Q,\text{output\_size}=m)$、$\tilde V=V\operatorname{softmax}(K^{\top}\tilde Q/\sqrt d)$，再取 $O^{\rm share}=\operatorname{FlashAttention}(Q,\tilde Q,\tilde V)$（Algorithm 1）[^src-2602-01219-mita]。这两步与 Agent attention 的 Stage 2（agent aggregation）、Stage 3（agent broadcast）逐项同型：同一个 $A$ 先作 query 从全部 $K,V$ 收信息，再作 key 被全部 query 读取。差别在 $A$ 的取法与后续：Agent attention 取 $\operatorname{Pooling}(Q)$，广播后即结束（Table 17 把池化记为 Dynamic）；MiTA 取分段平均池化，并在共享专家之外再加 top-$k$ 形变专家与 $N$-to-$m$ 路由[^src-2312-08874-agent-attention][^src-2602-01219-mita]。

MiTA 的分类表把 Agent attention 记为 Compression / 1 个共享 MLP / landmark testing / all-to-one；一手源对得上：agent token 直接取自 $Q$ 的池化、不训练，两段之外没有路由[^src-2312-08874-agent-attention][^src-2602-01219-mita]。上段「同源」判断是本 wiki 的对照分析，两篇源文均未互引为同一机制的来源。

## 边界

- COCO 训练用 80k 子集、35k 验证，不是完整的 118K/5K 划分（Appendix D.2）；Table 2 的表注按 $1280\times800$ 计 FLOPs，正文称 mAP 以 $(3,1333,800)$ 测试。
- 式 7 的复杂度只到 FLOPs 量级，softmax 归一化、池化、gather 的访存与 kernel 实现未计入，实测速度只有曲线（Fig. 5、Fig. 9）没有表格数值。
- 未给跨骨干的最优 agent token 数；与 GPViT、GRL 的差异只有定性论证，归结为 agent token 是否取自 query space（$A=f(Q)$）以及能否免训练迁移（Appendix A）[^src-2312-08874-agent-attention]。

## 相关页面

- [[source-2312-08874-agent-attention]] — 源摘要
- [[mita-attention]] — 共享专家与 landmark query 的后续机制
- [[fast-weight-scaling-taxonomy]] — 本方法在五维分类中的坐标
- [[cbsa]] · [[representative-token-extraction]] — 代表性 token 取法的另两条实例链
- [[linear-attention-unified-framework]] — 核函数级线性注意力统一
- [[adaptive-graph-agent-attention]] — 另一类 agent token 中介注意力（时空图）

[^src-2312-08874-agent-attention]: [[source-2312-08874-agent-attention]]
[^src-2602-01219-mita]: [[source-2602-01219-mita]]
