---
title: "SLiDE"
type: entity
tags:
  - time-series-forecasting
  - exogenous
  - koopman-operator
  - linear-dynamics
  - ood-generalization
  - neurips-2026
created: 2026-11-24
last_updated: 2026-11-27
source_count: 6
confidence: medium
status: active
---

# SLiDE

SLiDE（Structured Linear Dynamics for Forecasting with Exogenous Inputs）是一种面向外生变量时间序列预测的轻量级动力学架构[^src-slide]。多数外生预测模型依靠交叉注意力或深层 MLP 堆叠进行柔性融合，当测试期外生变量超出训练分布范围或驱动机制发生切换时，高自由度的非线性交互极易过拟合虚假相关而失效[^src-slide]。SLiDE 的核心巧思在于把潜空间演化收紧为一枚固定的分块线性转移算子 $W$ 与外生加性推力：历史编码阶段用它递推观测与外生信号来估计当前潜状态，未来预测阶段则复用完全同一套算子与外生编码器向前滚动，从而在不依赖复杂注意力的前提下压低模型自由度，专门提升分布外与跨驱动泛化能力[^src-slide]。

## 问题定位

带外生输入的时间序列预测需要同时利用历史内生观测、历史外生输入以及已知未来外生输入[^src-slide][^src-tide]。现有深度方案大致沿两条路径展开：一类以交叉注意力或双阶段图对齐做柔性融合（如 [[source-timexer|TimeXer]]、[[gcgnet|GCGNet]]），另一类则在 Koopman 控制视角下引入与潜状态耦合的双线性/状态相关控制项 $B(z,u)$（如 [[mmckm|MMCKM]]）[^src-slide][^src-timexer][^src-gcgnet][^src-mmckm]。这些高容量结构在平稳同分布基准上拟合能力强，但在外生变量出现超出训练范围的偏移（Out-of-Range, OOR）或驱动规则切换时，编码端与解码端各自积累的非线性自由度容易放大伪相关误差[^src-slide]。

SLiDE 提出反方向的约束策略：不再增强模块间的柔性非线性交互，而是强制历史编码与未来滚动共享同一线性动力系统，把非线性全部退让到输入嵌入层与输出汇总层[^src-slide]。

## 核心机制与三条结构硬约束

剥离论文术语，SLiDE 的前向计算完全由标准算子拼装而成：**跨步共享 MLP 嵌入 + 滞后不变分块线性循环编码 + 共享算子纯加性外生滚动 + 分块求和非线性解码**[^src-slide]。整个架构建立在三条显式结构硬约束之上：

1. **编码与滚动共享转移算子（$A_\psi = A_\theta = W$）**：历史状态编码期与未来滚动预测期严格共用同一块对角线性矩阵 $W$（含 $N=2$ 个子块），不再为编码器和预测器分别拟合独立转移矩阵[^src-slide]。
2. **历史与未来共享外生编码器（$B_\psi = B_\theta = \operatorname{MLP}_u$）**：过去步与预测步的外生变量经过同一份逐时间步映射 $\operatorname{MLP}_u$，确保外部驱动信号在整条时间轴上以统一尺度注入潜空间[^src-slide]。
3. **纯加性外生输入与输出端非线性替代 $C$**：放弃传统 Koopman-with-input 中随状态变化的控制项 $B(z, u)$，改以纯加性推力 $p_\tau = \operatorname{MLP}_u(u_\tau)$ 驱动潜状态；同时由于线性转移限制了中间态容量，输出端放弃线性观测矩阵 $C$，改为各子块分别经 $\operatorname{MLP}_{\hat{y}}$ 解码后求和[^src-slide]。

## 数据流与维度变换

> **维度符号**：$k$ 当前时刻（回看长度）；$H$ 预测步长；$d_y$ 目标变量维；$d_u$ 外生输入维；$d$ 潜状态维；$N=2$ 分块对角矩阵 $W$ 的子块数（每块维度 $d/N$）。以下均按单样本列向量记号书写，批维度与子块/投影形状按单样本约定推断标注为 `[INFERENCE]`[^src-slide]。

### Stage 1 · 跨步共享特征嵌入

读取历史目标观测 $y_\tau \in \mathbb{R}^{d_y}$（$\tau = 1..k$）及覆盖历史与未来的外生输入 $u_\tau \in \mathbb{R}^{d_u}$（$\tau = 1..k+H$），经两个跨时间步共享的 MLP 分别投影到 $d$ 维潜空间[^src-slide]。

$$\begin{aligned}
p_\tau &= \operatorname{MLP}_u(u_\tau) && \text{// } [d_u, 1] \xrightarrow{\text{跨历史与未来步共享 } \operatorname{MLP}_u} [d, 1]\ \text{——外生变量编码为加性推力 } (\tau=1..k+H)\ \text{[INFERENCE]} \\
d_\tau &= \operatorname{MLP}_y(y_\tau) && \text{// } [d_y, 1] \xrightarrow{\text{跨历史步共享 } \operatorname{MLP}_y} [d, 1]\ \text{——历史目标观测映射到同维潜空间 } (\tau=1..k)\ \text{[INFERENCE]}
\end{aligned}$$

### Stage 2 · 滞后不变线性循环编码初态

初始化零向量 $\pi_0 = \mathbf{0} \in \mathbb{R}^d$，用分块对角线性矩阵 $W = \operatorname{diag}(W^{(1)}, W^{(2)})$ 沿历史窗口递推至当前时刻 $k$，将终态直接作为未来滚动的初始潜状态 $z_k$[^src-slide]。

$$\begin{aligned}
\pi_\tau &= W\pi_{\tau-1} + d_\tau + p_\tau && \text{// } [d, \bcancel{d}]\times[\bcancel{d}, 1] + [d, 1] \to [d, 1]\ \text{——消去输入潜维，线性递推累计历史观测与外生推力} \\
z_k &= \pi_k = \sum_{\tau=0}^{k-1} W^\tau (d_{k-\tau} + p_{k-\tau}) && \text{// } [d, 1]\ \text{——闭式等价于滞后不变指数加权汇总，得滚动初态}
\end{aligned}$$

### Stage 3 · 线性加性外生滚动

令 $z_{k,0} = z_k$，沿预测步 $h = 1, \dots, H$ 复用 Stage 2 的线性算子 $W$ 与 Stage 1 产出的未来外生推力 $p_{k,h} = \operatorname{MLP}_u(u_{k+h})$ 向前滚动[^src-slide]。

$$\begin{aligned}
z_{k,h} &= W z_{k,h-1} + p_{k,h} && \text{// } [d, \bcancel{d}]\times[\bcancel{d}, 1] + [d, 1] \to [d, 1]\ \text{——沿用同一 } W \text{ 与纯加性外生推力线性演化第 } h \text{ 步潜状态}
\end{aligned}$$

### Stage 4 · 分块非线性输出汇总

将潜状态 $z_{k,h}$ 按分块对角结构切分为 $N=2$ 个独立子块 $z_{k,h}^b$，经输出端解码器 $\operatorname{MLP}_{\hat{y}}$ 映射回目标维度并求和[^src-slide]。

$$\begin{aligned}
\{z_{k,h}^b\}_{b=1}^{N} &= \operatorname{Split}_N(z_{k,h}) && \text{// } [d, 1] \xrightarrow{\text{沿潜维均分为 } N=2 \text{ 块}} N \times [d/N, 1]\ \text{——切分分块对角子空间表示 [INFERENCE]} \\
\hat{y}_{k,h} &= \sum_{b=1}^{N} \operatorname{MLP}_{\hat{y}}(z_{k,h}^b) && \text{// } [N, d/N] \xrightarrow{\text{逐块非线性解码后沿 } N \text{ 维求和}} [d_y, 1]\ \text{——以输出非线性替代传统线性矩阵 } C\ \text{[INFERENCE]}
\end{aligned}$$

该设计将潜空间维度压低且不设连续谱建模，计算复杂度随序列长度呈线性增长，推理延迟接近 TSMixer；代价是一旦外生输入 $u_\tau$ 缺失或存在时间错位，误差会经 $p_\tau$ 直接带入线性滚动轨迹[^src-slide]。

## 与既有工作的关系

SLiDE 在 [[koopman-linearization-for-forecasting|Koopman 线性化预测]] 与外生协变量预测两条谱系中形成以下明确位置：

- **同槽位互斥（vs [[koopman-linearization-for-forecasting|一般 Koopman 控制形式]]）**：库内既有 Koopman 预测模型均引入了与状态或外部流耦合的控制项 $B(z,u)$ 或残差精炼网络；SLiDE 是目前唯一仅用纯加性外生推力、且强制编码与滚动共享同一线性矩阵 $W$ 的预测口径[^src-slide][^src-k2vae][^src-mmckm]。
- **同构/特例（vs [[mmckm|MMCKM]] / [[intent-discriminator-koopman|Intent Discriminator]]）**：MMCKM 在交通微观层面采用由 5 模态参数有界 Koopman 算子组成的 MoE 家族，并经 CrossAttention 注入宏观流控制 $B_z u_t$；SLiDE 可视为该状态相关多模态 Koopman 族的线性加性简化特例——以单枚固定 $W$ 与纯加性输入换取分布偏移下的低过拟合自由度[^src-slide][^src-mmckm]。
- **跨槽位正交（vs [[k2vae|K²VAE]]）**：K²VAE 面向无外生变量的长期概率预测，结合 one-step eDMD 局部算子、全局 $K_{glo}$ 与 KalmanNet 不确定性建模执行 VAE 一步生成；SLiDE 则放弃概率变分与 Kalman 滤波，专注于未来已知外生输入下的确定性点预测与 OOR 稳健性[^src-slide][^src-k2vae]。
- **同槽位互斥（vs [[tide|TiDE]]）**：TiDE 将历史目标与全部协变量展平送入 dense residual MLP 编码，再经 [[temporal-decoder|Temporal Decoder]] 逐预测步并入未来协变量；SLiDE 把时序演化抽离为共享分块线性循环递推 $W$，外生信号仅以加性推力进入潜状态演化[^src-slide][^src-tide]。
- **同槽位互斥（vs [[source-timexer|TimeXer]] / [[gcgnet|GCGNet]]）**：TimeXer 用 patch-wise 自注意力和 variate-wise 交叉注意力融合外生变量，GCGNet 用 VAE 粗预测配合联合图对齐与 GCN 精炼；SLiDE 去掉注意力与图精炼，在同生态位外生基准上以更低计算开销取得更高平均排名，且面向 [[heterogeneous-covariates|异构时间角色协变量]]（future-known、past-only 与 static）给出实测对比[^src-slide][^src-timexer][^src-gcgnet]。

| 模型 | 状态转移形式 | 外生 / 控制注入 | 编码与滚动是否共享动力学 | 输出解码与预测性质 |
|---|---|---|---|---|
| **SLiDE** | 固定 $N=2$ 分块线性矩阵 $W$ | 纯加性 $\operatorname{MLP}_u(u_\tau)$ | 是（$A_\psi=A_\theta=W$，$B_\psi=B_\theta$） | 分块 $\operatorname{MLP}_{\hat{y}}$ 求和；确定性点预测[^src-slide] |
| [[k2vae\|K²VAE]] | eDMD $K_{loc}+K_{glo}$ + KalmanNet | 无外生变量（非线性残差作控制） | 否（KoopmanNet + KalmanNet 两段） | VAE 采样 + 双 MLP 解码；概率分布预测[^src-k2vae] |
| [[mmckm\|MMCKM]] | 5 模态参数有界 Koopman MoE | CrossAttention 状态耦合控制 $B_z u_t$ | 否（history-free 快照编码后多步滚动） | 线性/MLP 解码；交通微观轨迹预测[^src-mmckm] |
| [[tide\|TiDE]] | Dense residual MLP + 全局线性残差 | 展平编码 + Temporal Decoder 合并 | 否（展平编码器 + 多步解码器） | 残差 MLP 逐维输出；确定性点预测[^src-tide] |
| [[source-timexer\|TimeXer]] | Patch 自注意力 | 跨变量交叉注意力 | 否（直接多步输出头） | 线性映射头；确定性点预测[^src-timexer] |
| [[gcgnet\|GCGNet]] | VAE 粗预测 + 多层 GCN 图精炼 | 联合图对齐合并内生/外生 patch | 否（生成器 + 对齐器 + 精炼器） | 展平线性头；确定性点预测[^src-gcgnet] |

## 证据

论文在 EPF-5、Favorita 与 Bouc-Wen 基准及消融设置上报告了以下实证结果（均为作者报告口径）[^src-slide]：

1. **定位与主结论**：论文报告 SLiDE 在电力价格 5 个市场、零售与合成非线性系统共 8 项任务上平均排名 1.87 居首（11 个指标实例中 8 次进入前二），推理开销接近 TSMixer 且明显低于 [[source-timexer|TimeXer]] 与 [[gcgnet|GCGNet]]；优势集中在偏移设置——EPF-DE 越界子集较最强基线降错约 20%，跨驱动迁移较次优降错约 30%[^src-slide]。
2. **真实涨点归因（Ablation）**：掉点最狠的是线性转移约束——将线性转移 $W$ 替换为两层 ReLU MLP 后，Bouc-Wen 同分布精度反而更好，但跨驱动迁移大幅劣化、Favorita 越界子集落后约 18% RMSLE，说明线性递推专门服务分布偏移泛化而非同分布精度；换成普通 MLP 编码器则全面退化（平均排名 1.87 → 2.47）；「编码与滚动共享算子」这一核心约束本身未被单独消融[^src-slide]。
3. **证据缺口与负结果**：偏移评估仅覆盖每数据集样本量有限的越界子集与单一合成迁移（论文自述）；合成同分布 multisine 任务上系统辨识系基线一度领先，但其整体平均排名最靠前也仅 5.8，未构成同生态位威胁；论文未报告概率校准类指标[^src-slide]。

## 相关页面

- [[source-slide]] — 论文源摘要
- [[heterogeneous-covariates]] — 跨槽位正交：future-known / past-only / static 异构时间角色协变量设定
- [[ood-generalization]] — 跨槽位正交：时序分布外泛化与偏移评估语境（其余 Koopman 与外生基线的详细关系见上文「与既有工作的关系」）

[^src-slide]: [[source-slide]]
[^src-k2vae]: [[source-k2vae]]
[^src-mmckm]: [[source-mmckm]]
[^src-tide]: [[source-tide]]
[^src-gcgnet]: [[source-gcgnet]]
[^src-timexer]: [[source-timexer]]
