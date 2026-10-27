---
title: "Hyper-Connections (HC)"
type: technique
tags:
  - residual-connections
  - macro-design
  - large-language-models
  - deep-learning
created: 2026-06-22
last_updated: 2026-10-27
source_count: 2
confidence: high
status: active
---

# Hyper-Connections (HC)

Hyper-Connections（HC）是 ByteDance Seed 提出的残差连接替代方案：把单路隐向量复制成 $n$ 份并行的超隐向量，每层用 $(n+1)\times(n+1)$ 连接矩阵决定层从哪些流取输入、把输出写回哪些流，以及流与流之间如何混合；连接强度由网络静态学习或由输入动态生成[^src-hyper-connections]。

## 动机：跷跷板效应

Pre-Norm 保持梯度顺畅，但各层输出等权叠加易引发表示坍缩，相邻层特征余弦相似度居高（Fig. 3）；Post-Norm 抑制坍缩，但信号逐层被归一化缩放，导致浅层连接衰减，重新引入梯度消失。这对矛盾被称为跷跷板效应（seesaw effect），根源在于 Pre/Post-Norm 预先写死了层输入与输出的连接强度；HC 则让该强度可学习[^src-hyper-connections]。$n=1$ 时跷跷板仍在：数学上各前层连接只能同强同弱缩放，实验确认 $n=1$ 劣于基线；$n>1$ 才允许不同前层拥有不同连接，并进一步支持层间串并行重排[^src-hyper-connections]。

## 单层形式

设第 $k$ 层输入为超隐矩阵 $\mathbf{H}=(\mathbf{h}^1\ \dots\ \mathbf{h}^n)^\intercal\in\mathbb{R}^{n\times d}$，$d=d_{model}$，$n$ 为 expansion rate（扩展率）。连接矩阵按行分块（公式 1）：

$$\mathcal{HC}=\begin{pmatrix}\mathbf{0}^{1\times1} & \mathbf{B}\\ \mathbf{A_m} & \mathbf{A_r}\end{pmatrix}\in\mathbb{R}^{(n+1)\times(n+1)},\qquad \mathbf{B}\in\mathbb{R}^{1\times n},\ \ \mathbf{A_m}\in\mathbb{R}^{n\times1},\ \ \mathbf{A_r}\in\mathbb{R}^{n\times n}$$

对层函数 $\mathcal{T}$（Transformer 中即 attention 或 FFN），单层更新（公式 2）为：

$$\hat{\mathbf{H}}=\mathcal{HC}(\mathcal{T},\mathbf{H})=\mathbf{B}^\intercal\,\mathcal{T}(\mathbf{H}^\intercal\mathbf{A_m})^\intercal+\mathbf{A_r}^\intercal\mathbf{H}$$

三个分块各司其职（公式 3–5）[^src-hyper-connections]：

$\mathbf{A_m}$ 把 $n$ 条流加权求和为层输入向量 $\mathbf{h}_0$（公式 3）；$\mathbf{A_r}$ 把 $\mathbf{H}$ 映射为新超隐矩阵 $\mathbf{H'}=\mathbf{A_r}^\intercal\mathbf{H}$（公式 4）；$\mathbf{B}$ 把层输出 $\mathcal{T}\mathbf{h}_0$ 按权重写回 $n$ 条流（公式 5）。

连接矩阵解耦为两个组件（公式 6–7）：depth-connections $\mathcal{DC}=\begin{pmatrix}\mathbf{B}\\ \mathrm{diag}(\mathbf{A_r})\end{pmatrix}\in\mathbb{R}^{2\times n}$ 是每条流各自的「输出权重 + 输入权重」，即残差连接的直接推广；width-connections $\mathcal{WC}=(\mathbf{A_m}\ \ \mathbf{A_r})\in\mathbb{R}^{n\times(n+1)}$ 负责同层各流间的横向交换[^src-hyper-connections]。每个 Transformer 层配有两个 HC 模块（attention、FFN 各一，Appendix B）；网络末端将最后的超隐矩阵按行求和为单向量，再经 norm 与 unembedding（Algorithm 1）[^src-hyper-connections]。

## 数据流与维度变换

**维度符号**：$n$ 为 expansion rate（并行流数，消融取 1–8，默认 4）；$d=d_{model}$（OLMo-1B 取 2048）；$L$ 为层数；各子层配一个 HC 模块。形状记 $[\text{行}, \text{列}]$，与论文一致。

### Stage 1 · 复制展开

$$\begin{aligned}
\mathbf{H}_0 &= (\mathbf{h}_0\ \mathbf{h}_0\ \dots\ \mathbf{h}_0)^\intercal && \text{// } [d] \xrightarrow{\text{复制 } n \text{ 份}} [n, d]\ \text{——单路嵌入展开为 } n \text{ 条内容相同的并行残差流}
\end{aligned}$$

### Stage 2 · width connection 混合出层输入

$$\begin{aligned}
\mathbf{h}_0^\intercal &= \mathbf{A_m}^\intercal\mathbf{H} && \text{// } [1, \bcancel{n}]\times[\bcancel{n}, d]\to[1, d]\ \text{——按 } \mathbf{A_m} \text{ 加权求和 } n \text{ 条流，得到当前子层的单流输入}
\end{aligned}$$

### Stage 3 · 动态生成连接系数（DHC）

$$\begin{aligned}
\overline{\mathbf{H}} &= \texttt{norm}(\mathbf{H}) && \text{// } [n, d]\to[n, d]\ \text{——超隐矩阵归一化（论文写 } \texttt{norm}\text{；附录 G 与官方代码用 LayerNorm，OLMo 中无参数、OLMoE 中带 } d \text{ 参数）}\\
\mathbf{B}(\mathbf{H}) &= s_\beta\circ\tanh(\overline{\mathbf{H}}\mathbf{W}_\beta)^\intercal+\mathbf{B} && \text{// } [n, \bcancel{d}]\times[\bcancel{d}, 1]\xrightarrow{\text{转置}}[1, n]\ \text{——每条流的内容各打一个分，作为该流的层输出权重}\\
\mathbf{A_m}(\mathbf{H}) &= s_\alpha\circ\tanh(\overline{\mathbf{H}}\mathbf{W}_m)+\mathbf{A_m} && \text{// } [n, \bcancel{d}]\times[\bcancel{d}, 1]\to[n, 1]\ \text{——每条流各出一个标量，合成「读哪条流」的权重}\\
\mathbf{A_r}(\mathbf{H}) &= s_\alpha\circ\tanh(\overline{\mathbf{H}}\mathbf{W}_r)+\mathbf{A_r} && \text{// } [n, \bcancel{d}]\times[\bcancel{d}, n]\to[n, n]\ \text{——流 } i \text{ 对流 } j \text{ 的混合系数由流 } i \text{ 的内容生成}
\end{aligned}$$

三个投影共享同一管线（公式 10–13）：归一化 → 线性投影 → tanh → 乘可学习缩放因子 $s$ → 加静态矩阵；SHC 则直接使用静态矩阵 $\mathbf{B},\mathbf{A_m},\mathbf{A_r}$。训练时静态分量不加 weight decay，动态分量施加（§4）[^src-hyper-connections]。

### Stage 4 · depth connection 写回全部流

$$\begin{aligned}
\hat{\mathbf{H}} &= \mathbf{B}^\intercal(\mathcal{T}\mathbf{h}_0)^\intercal && \text{// } [n, \bcancel{1}]\times[\bcancel{1}, d]\to[n, d]\ \text{——层输出按 } \mathbf{B} \text{ 广播写回 } n \text{ 条流（外积）}\\
&\quad+\mathbf{A_r}^\intercal\mathbf{H} && \text{// } [n, \bcancel{n}]\times[\bcancel{n}, d]\to[n, d]\ \text{——}\mathbf{A_r} \text{ 在流间横向混合，承载 width-connections}
\end{aligned}$$

子层本体 $\mathcal{T}$ 的输入输出均为单流 $[1, d]$，FLOPs 与标准 Transformer 相同；HC 仅在外围做小矩阵系数运算。

### Stage 5 · 末端汇合

$$\begin{aligned}
\mathbf{h}_L &= \textstyle\sum_{i=1}^{n}\mathbf{H}_{L,i,:} && \text{// } [n, d] \xrightarrow{\text{沿流维求和}} [d]\ \text{——} n \text{ 条流求和回落为单向量，再过 norm 与 unembedding（Algorithm 1）}
\end{aligned}$$

### 初始化：严格等价 Pre-Norm 残差

动态投影 $\mathbf{W}_\beta, \mathbf{W}_m, \mathbf{W}_r$ 初始化为 0（动态项归零），静态矩阵初始化为（公式 14）：

$$\begin{pmatrix}\mathbf{0}^{1\times1} & \mathbf{1}^{1\times n}\\ \mathbf{e}_{k \bmod n} & \mathbf{e}_{n\times n}\end{pmatrix}$$

即第 $k$ 层读流 $k \bmod n$，输出以全 1 权重写回所有流，流间为恒等映射；在此初始化下，HC 严格等价于 Pre-Norm 残差连接（$n$ 份相同副本）[^src-hyper-connections]。缩放因子 $s_\alpha, s_\beta$ 初始为 0.01（官方代码 Algorithm 2）。为使末端求和后的输出 std 与原模型一致，所有层输出模块（FFN 第二线性、attention 输出投影）的权重 std 在初始化时乘 $\sqrt{n}$（§4 Implementation）[^src-hyper-connections]。

## 残差连接的统一视角

Pre-Norm 与 Post-Norm 都是非训练的 HC 特例（§3.1）。取 $n=1$（公式 15–16）：

$$\mathcal{HC}^{PreNorm}=\begin{pmatrix}0 & 1\\ 1 & 1\end{pmatrix},\qquad \mathcal{HC}^{PostNorm}=\begin{pmatrix}0 & \tfrac{1}{\sqrt{\sigma_i^2+\sigma_o^2+2\sigma_{io}}}\\ 1 & \tfrac{1}{\sqrt{\sigma_i^2+\sigma_o^2+2\sigma_{io}}}\end{pmatrix}$$

Post-Norm 的权重依赖层输入/输出的方差与协方差，是不可训练的统计量；HC 将矩阵放宽为可训练甚至输入依赖，完整推导见原文 Appendix G[^src-hyper-connections]。

## 序贯-并行对偶

HC 矩阵的不同取值对应不同层间拓扑（§3.2，$n=2$）：公式 17 使各层严格串行，depth connection 退化为普通残差；奇数层与偶数层分别取公式 18–19 的矩阵时，相邻两层并行执行（parallel transformer blocks, PTB）。学得的矩阵也可落在串行与并行之外的软混合上；静态 HC 的拓扑在训练后固定，DHC 的拓扑则逐 token 动态变化（一般推导见 Appendix H）[^src-hyper-connections]。

## 规模与开销（Appendix B）

每层两个 HC 模块：SHC 引入 $n(n+2)\times 2L$ 个参数（OLMo-1B-SHC×4 仅 768 个），DHC 约引入 $d_{model}(n+2)\times 2L$ 个。OLMo-1B-DHC×4 参数 +0.033%（Table 7），FLOPs +0.200%（Table 8，2.3536→2.3583 G/token）；7B 为 +0.023% / +0.147%。激活显存增加 $2nsbd_{model}L$（$b$ batch、$s$ 序列长），前向丢弃并在反向重算可降为 $nsbd_{model}L$，KV cache 不受影响；8 卡实测 OLMo-1B-DHC×4 显存 41.11→51.86 GB（+26.1%），OLMoE-1B-7B 为 +9.7%（Table 9）[^src-hyper-connections]。

## 实验证据

语言模型均为 500B tokens 的 OLMo/OLMoE 复现设置，默认方法为含 tanh 的 DHC×4[^src-hyper-connections]。

| 实验 | 结果 | 出处 |
|---|---|---|
| $n$ 消融（1B，Table 1） | baseline V2 loss 2.811 / V3 2.544 / 下游 62.5；DHC×4 为 2.781 / 2.514 / 63.8；DHC×8 W/O tanh 的 loss 最低（V2 −0.034、V3 −0.029）；DHC×1（2.819 / 62.3）劣于 baseline | §4.1 |
| SHC vs DHC（Table 2） | $n=2$ 两者相近（下游 63.4 对 63.0）；$n=4$ DHC 更好（V2 loss 2.781 对 SHC×4 的 2.791） | §4.1 |
| WC/B 可训练性（Table 3，DHC×4） | 冻结 WC：V2 +0.021、V3 +0.017；冻结 B：影响小（V2 2.783）——WC 不可训练的损失明显大于 B 不可训练 | §4.1 |
| 对比 ResiDual / AltUp（Table 4，均 $n=2$） | ResiDual 2.825 / 下游 62.0，AltUp×2 2.827 / 62.4：训练早期有增益、随后被 baseline 反超；同 $n$ 的 DHC×2 为 2.802 / 63.0 | §4.2 |
| 7B 密集模型（Table 5） | V2 loss 2.581→2.559、PPL 14.316→14.023、下游 70.1→71.0，FLOPs 13.36→13.38 G/token；baseline 频现 loss spike，DHC 全程无 spike（Fig. 6） | §4.3 |
| OLMoE-1B-7B（Table 6） | 训练损失 −0.027、C4-en 验证 loss −0.028；ARC-C 41.8→47.8、MMLU Var 38.5→39.7、ARC-E 72.8→76.7、BoolQ 65.4→68.5；达到相同下游水平约省一半 token（Fig. 9，作者报告），收敛快 1.8×（§1、Fig. 1） | §4.4 |
| DiT 图像生成（Table 10，ImageNet 256，cfg 1.5） | DiT-XL/2（FP16 + QK-Norm）FID 2.36 → 加 SHC×2 后 2.18，与参数多 50% 的 DiT-1B/2（983M，FID 2.13）相当 | App. E.1 |
| ViT 分类（Table 11，$n=2$，300 epoch） | Base（85M）：76.38 → SHC 77.60 / DHC 77.26；Large（307M）：77.25 → SHC 78.38 / DHC 79.94 | App. E.2 |

OLMo-1B-DHC×4 跨层展开连接矩阵的可视化（§4.5、Appendix F）显示，学到的连接呈 Λ 形——近层连接强、随距离衰减（Post-Norm 式），同时底层输出被后续层高频访问（Pre-Norm 式），二者在同一矩阵中自由混合；输入词嵌入被除末层外的多数层使用，而产出模型输出的末层基本丢弃嵌入分量；PTB 式并行模式自发涌现（如 layer 11 对 layer 12 输入贡献极小）；attention 输出的长期连接显著少于 FFN，形态接近 two-hop residual。HC×1 会出现与其他层完全断连的「废层」，且 $n=1$ 在数学上不允许 Λ 形连接[^src-hyper-connections]。

## 边界

$n=1$ 劣于基线，$n>1$ 是收益前提（Table 1；Appendix F）[^src-hyper-connections]。ViT 多 epoch 训练增益随进度递减，作者归因于重复遍历同一数据时额外容量收益递减（Fig. 11）[^src-hyper-connections]。下游均值剔除了相邻 checkpoint 波动超过 20% 的高方差数据集（Table 13 灰色标注）[^src-hyper-connections]。实验止于 500B tokens、密集 ≤7B 与激活 1.3B 的 MoE，未覆盖更大规模下无约束 $\mathbf{A_r}$ 的稳定性；mHC 随后报告 HC 复合残差映射增益峰值近 3000 并引发损失尖峰，见 [[manifold-constrained-hyper-connections]][^src-mhc-manifold-constrained-hyper-connections]。

## 与 mHC 的关系

[[manifold-constrained-hyper-connections|mHC]] 保留 HC 的多流拓扑，把流间混合矩阵投影到双随机矩阵流形以恢复恒等映射。mHC 将 HC 的 $\mathbf{A_m} / \mathbf{A_r} / \mathbf{B}$ 改记为 $H^{pre} / H^{res} / H^{post}$（其中 $\mathbf{A_m}$ 与 $H^{pre}$ 互为转置），动态分量改用 RMSNorm 实现；记号与术语沿用 HC 原文[^src-hyper-connections][^src-mhc-manifold-constrained-hyper-connections]。

谱系后续：[[attnres|Attention Residuals]] 把 HC 一类多流递归统一为深度维线性注意力，并以深度维 softmax 注意力直接检索各层输出。

[^src-hyper-connections]: [[source-seed-2024-hyper-connections]]
[^src-mhc-manifold-constrained-hyper-connections]: [[source-mhc-manifold-constrained-hyper-connections]]
