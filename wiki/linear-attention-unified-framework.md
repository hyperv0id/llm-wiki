---
title: "Mamba ↔ Linear Attention Unified Framework"
type: concept
tags:
  - mamba
  - linear-attention
  - unified-framework
  - architecture-analysis
created: 2026-05-08
last_updated: 2026-10-27
source_count: 6
confidence: medium
status: active
---

# Mamba ↔ Linear Attention Unified Framework

Han et al. (NeurIPS 2024) 提出的统一框架，将 Mamba（选择性状态空间模型）和线性注意力 Transformer 纳入同一数学形式中，揭示了 Mamba 本质上是线性注意力的一个变体[^src-demystify-mamba-linear-attention-2024]。

## 数学对应关系

### 线性注意力的循环形式

线性注意力使用核函数 $\phi$ 替代 softmax，其循环形式为[^src-demystify-mamba-linear-attention-2024]。该递推的原始出处是 Katharopoulos et al. 2020 的因果掩码线性化（式 10–12 与式 18–19，记号为 $s_i=s_{i-1}+\phi(k_i)v_i^{\top}$、$z_i=z_{i-1}+\phi(k_i)$），机制推导与实验见 [[linear-attention-rnn-form]][^src-katharopoulos-2020-linear-attention]：

$$S_i = S_{i-1} + K_i^\top V_i, \quad Z_i = Z_{i-1} + K_i^\top$$
$$y_i = \frac{Q_i S_i}{Q_i Z_i}$$

### Mamba 的 SSM 形式

Mamba 的选择性 SSM 为[^src-demystify-mamba-linear-attention-2024]：

$$h_i = \widetilde{A}_i \odot h_{i-1} + B_i(\Delta_i \odot x_i)$$
$$y_i = C_i h_i + D \odot x_i$$

展开为累积形式：

$$y_i = C_i \sum_{j=1}^i (\prod_{k=j+1}^i \widetilde{A}_k) B_j (\Delta_j \odot x_j) + D \odot x_i$$

### 组件对应表

| Mamba 组件 | 线性注意力对应 | 功能描述 |
|-----------|---------------|---------|
| $C_i$ | $Q_i$ | 输出/查询投影 |
| $B_j$ | $K_j^\top$ | 输入/键投影 |
| $\Delta_j \odot x_j$ | $V_j$ | 值（带输入门控） |
| $\prod_{k=j+1}^i \widetilde{A}_k$ | 隐式恒等（1） | 遗忘门（线性注意力中无） |
| $D \odot x_i$ | 无对应 | 捷径残差连接 |
| $Q_i Z_i$ 分母 | Mamba 中缺失 | 注意力归一化 |

## 六项关键差异

基于统一框架，论文识别出 Mamba 与线性注意力的六项差异，并通过消融实验量化其影响[^src-demystify-mamba-linear-attention-2024]：

| 差异 | 对准确率影响 | 对吞吐量影响 |
|------|-------------|-------------|
| 输入门 $\mathbf{\Delta}_i$ | +0.9% | 1152→106 im/s |
| 遗忘门 $\widetilde{\mathbf{A}}_i$ | +0.8% | 1152→743 im/s |
| 捷径连接 $D \odot x_i$ | +0.2% | 几乎不变 |
| 无归一化 | **-5.2%** | — |
| 单头 | 边际/负面 | — |
| 块设计 | **+1.8%** | — |

## 核心洞见

1. **遗忘门可被位置编码替代**：遗忘门提供局部偏置和输入依赖的位置信息，可通过 LePE、CPE、RoPE 等位置编码实现，从而避免循环计算[^src-demystify-mamba-linear-attention-2024]
2. **归一化至关重要**：移除归一化导致 -5.2% 的灾难性下降，说明 Mamba 通过其他机制（如遗忘门的衰减效应）间接补偿了归一化
3. **块设计贡献最大**：修改的块设计（+1.8%）是 Mamba 在视觉任务中成功的最重要因素

在交通预测中，[[stgformer|STGformer]]（arXiv:2410.00385，2024）将此类分解内积线性注意力与 SGC 式图传播的多阶输出组合为统一时空注意力模块，论文给出该模块 O(N+T) 的时间/内存复杂度，并以单层结构替代堆叠时空注意力，见 [[stg-attention|STG-Attention]][^src-stgformer]。

## 与 fast-weight scaling 视角的分工

本页是公式级统一：Mamba 的选择性 SSM 与线性注意力被对齐成同一递推，差别落在六项可消融组件上[^src-demystify-mamba-linear-attention-2024]。另一条统一路线（MiTA，arXiv:2602.01219）不写等价公式，而是把 full attention 读成宽度等于序列长度的两层 fast-weight MLP，再按「压缩／路由」两条轴给各方法坐标；在这个坐标系里线性注意力只是「压缩成 1 个共享线性层、all-to-one」的一格[^src-2602-01219-mita]。

两者不冲突、也不互相蕴含：本页回答「两个方法差在哪几项」，五维分类回答「各方法把 $N$ 宽度降在哪里」。引用时按问题选口径，不要把「线性注意力在分类表里占一格」误读成对 Mamba 等价性结论的削弱。

## 核函数的另一类取法：中介 token

本页的对应关系以「已给定的 $\phi$ 之间如何互换」为对象。Agent attention（arXiv:2312.08874）给的是 $\phi$ 本身的参数化：把核映射定义为 $\phi_q(Q)=\sigma(QA^{\top})$、$\phi_k(K)=\sigma(AK^{\top})^{\top}$，其中 $A\in\mathbb{R}^{n\times C}$ 是 agent token，于是 $\sigma(QA^{\top})\sigma(AK^{\top})V$ 既是两段 softmax 的复合，也是核化线性注意力（式 4）[^src-2312-08874-agent-attention]。按结合律先算 $\phi_k(K)^{\top}V$ 得到 $[n,C]$ 的中间量，正是该方法的 agent aggregation 一步，随后左乘 $\phi_q(Q)$ 完成 broadcast；$n$ 即核映射维数，中介宽度 $n\ll N$ 时复杂度为 $\mathcal{O}(Nnd)$。与 Mamba 一侧的对应关系不同，此处没有遗忘门一类的递推修正项，全部差异集中在 $\phi$ 由何种 token 定义[^src-2312-08874-agent-attention]。

## 相关页面

- [[mamba|Mamba]] — 选择性状态空间模型
- [[mila|MILA]] — 基于此框架的线性注意力模型
- [[forget-gate-in-sequential-models|遗忘门在序列模型中的作用]]
- [[mamba-block-design|Mamba 块设计]]
- [[linear-attention-bias|线性注意力偏置（ALiBi）]]
- [[urbanpg|UrbanPG]] — STCA 模块直接使用 Performers 随机特征映射线性注意力，是该框架在 STGNN 领域的最新实例 (AAAI 2026)
- [[stg-attention|STG-Attention]] — STGformer 的统一时空线性注意力模块：分解内积 + 图传播多阶输出 (arXiv 2024)
- [[fast-weight-mlp-view-of-attention|fast-weight MLP 视角]] — 另一条注意力统一路线：结构级坐标而非等价公式
- [[fast-weight-scaling-taxonomy|fast-weight scaling 分类法]] — 五维坐标，线性注意力在其中占「压缩 + all-to-one」一格
- [[agent-attention|Agent Attention]] — 以 agent token 参数化核函数的线性注意力实例，两段 softmax 与 aggregate/broadcast 同序
- [[performer|Performer]] — 核随机特征路线的代表机制：softmax 核的正随机特征无偏估计 + 正交化降方差，线性注意力时间 O(Lrd)（ICLR 2021）[^src-performer]
- [[positive-random-features|PRF]] — Performer 论文中 softmax 核估计的数学核心：trig 特征在小核值区 MSE 发散、正特征收敛
- [[linear-attention-rnn-form]] — 上述循环形式的原始出处（Katharopoulos 2020）：核化推导、elu+1、因果前缀和与 RNN 重写

[^src-demystify-mamba-linear-attention-2024]: [[source-demystify-mamba-linear-attention-2024]]
[^src-stgformer]: [[source-stgformer]]
[^src-performer]: [[source-performer]]
[^src-2602-01219-mita]: [[source-2602-01219-mita]]
[^src-2312-08874-agent-attention]: [[source-2312-08874-agent-attention]]
[^src-katharopoulos-2020-linear-attention]: [[source-katharopoulos-2020-linear-attention]]
