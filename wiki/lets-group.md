---
title: "Let's Group (SGL): Plug-and-Play SubGraph Learning"
type: technique
tags:
  - spatiotemporal
  - traffic-forecasting
  - subgraph-learning
  - memory-efficiency
  - graph-neural-networks
  - ijcai-2025
created: 2026-08-29
last_updated: 2026-08-29
source_count: 4
confidence: medium
status: active
---

# Let's Group (SGL): Plug-and-Play SubGraph Learning

**Let's Group** 是 Weng 等人发表于 IJCAI-25 的即插即用子图学习方法（SubGraph Learning, SGL），目标是降低时空图神经网络（STGNN）的 GPU 内存开销而不明显损失预测性能[^src-lets-group]。方法由两个模块组成：SubGraph Partition Module（SGPM）用一组可学习记忆向量按特征相似度把时空图划分为多个小子图；SubGraph Feature Aggregation Module（SGFAM）再把同一节点跨子图的特征平均聚合[^src-lets-group]。作者报告：在四个不同规模的交通数据集上，SGL 变体保持与原模型相当的预测性能，平均 GPU 内存开销最高降低 56.4%（摘要与贡献节，未说明该平均的统计口径）[^src-lets-group]。

## 问题：STGNN 的内存开销随节点数增长

论文的出发点是 STGNN 的内存与计算瓶颈（Sec. 1、Sec. 3.6）。现有 STGNN 通过动态图构造或注意力机制提取空间特征，复杂度因相关矩阵构造与特征加权而与节点数 N 高度耦合，图卷积与注意力机制均为 O(N²)，内存开销随 N 显著增长，论文认为这限制了实际应用[^src-lets-group]。

一个直观方案是把时空图切成小子图分别提取特征再聚合，但论文指出，现有的子图划分方法是静态的（Sec. 1–2，论文自述）：PatchSTG 按节点地理坐标划分，FCGCN 用 Louvain 算法按拓扑关系划分，LarSTL（Wang et al., IJCAI-24）用 METIS 加地理坐标划分子图用于持续学习；论文认为这类静态划分无法捕获复杂时空依赖[^src-lets-group]。

## 机制

### 常规时空图特征提取的三步

论文把现有 STGNN 的空间特征提取概括为三步（Sec. 3.2, Fig. 1）[^src-lets-group]：

1. 节点特征提取 $H = F_{in}(X_{in})$：把原始输入映射到隐空间（如 MLP）；
2. 相关矩阵构造 $A = G(H)$：用动态图构造或注意力分数矩阵表示节点间关系；
3. 特征加权 $H_{out} = F_{out}(AH)$：用相关矩阵增强节点表示。

复杂度瓶颈在第 2、3 步：论文称图卷积与注意力机制的复杂度均为 O(N²)（Sec. 3.6）[^src-lets-group]。

### SGPM：记忆向量锚点的子图划分

SGPM 置于特征提取函数 $F_{in}$ 之后（Sec. 3.3, Fig. 2）。用 M 个随机初始化的可学习记忆向量 $P = [P_1, \dots, P_M] \in \mathbb{R}^{M \times D}$ 作为锚点（摘要称 learnable memory vectors）[^src-lets-group]：

- 相似度：$w_i = \mathrm{softmax}(H P_i^T) \in \mathbb{R}^{N \times 1}$，度量记忆向量 $P_i$ 与各节点特征的相似度（Eq. 5）；
- 划分：$idx_i = \mathrm{arg\ topK}(w_i)$，取与 $P_i$ 相似度最高的 K 个节点组成一个子图（Eq. 6–7）。

论文的假设是「高度相关的节点往往有相似特征」（Sec. 3.3）[^src-lets-group]。每个记忆向量独立选 top-K，因此节点可以同时出现在多个子图中（共 M×K 个索引，子图之间有重叠）[^src-lets-group]。

### 子图特征提取

每个子图独立构造相关矩阵 $A_i^{sub} = G(H_i^{sub}) \in \mathbb{R}^{K \times K}$，所有子图共享同一个 G()，再做特征加权 $H_{out}^{sub} = F_{out}(A^{sub} H^{sub})$（Eq. 8–9, Sec. 3.4）[^src-lets-group]。

### SGFAM：跨子图特征平均

划分带来的新问题是特征冗余：同一节点出现在多个子图会得到多份特征。SGFAM 置于特征加权层之后（Sec. 3.5, Fig. 3）：按子图索引收集同一节点跨子图的全部特征 $H^i$，取平均作为最终特征——$H_{out}^i = \mathrm{sum}(H^i)/\mathrm{count}(H^i)$，count 为 0 时输出 0（Eq. 10–12）[^src-lets-group]。

### 复杂度

论文的复杂度分析（Sec. 3.6）：传统方法为 O(N²)；SGL 由三部分组成——子图划分 O(NM+MK)、相关矩阵构造 O(K²)（各子图并行计算）、特征聚合 O(MK+N)，总体 O(NM+MK+K²+N)。论文称 M 相对 K、N 小得多可视为常数，简化为 O(N+K²)；当 K 远小于 N 时显著低于传统方法[^src-lets-group]。

## 实验证据

### 设置

数据集为四个交通网络数据集（Table 1）：PEMS03（358 节点）、PEMS04（307）、PEMS07（883）、PEMS08（170）；论文将其分为小规模（PEMS08）、中规模（PEMS03/04）、大规模（PEMS07）。按 6:2:2 时序划分训练/验证/测试，单张 4090 GPU，PEMS07 batch size 16、其余 64（Sec. 4）[^src-lets-group]。

backbone 共 8 个模型行（Sec. 4）：并行计算模型 GMAN、STWave-full GAT、STWave-ESGAT、[[staeformer|STAEformer]]、DGCNet(P)；串行计算模型 DGCRN、DDGCRN、DGCNet(R)。SGL 变体把 SGPM 与 SGFAM 接入各 backbone 的空间特征提取模块，训练配置（优化器、最大 epoch、early stop）与原模型一致。M/K 设置（Table 2）：PEMS03 M=4、K=100；PEMS04 M=4、K=80；PEMS07 M=10、K=100；PEMS08 M=4、K=50（DGCRN 与 DGCNet(R) 为 60）。

### 性能与内存（Q1）

作者报告各 backbone 及其 SGL 变体预测性能相当，而内存开销显著下降（Table 3）[^src-lets-group]：

- 摘要与贡献节口径：平均 GPU 内存开销最高降低 56.4%（论文未说明平均口径）；
- Sec. 4.1：小规模 PEMS08 上 DDGCRN-SGL 内存较 DDGCRN 降 18.2%，大规模 PEMS07 上降 60.5%（Table 3 中对应 2.85→2.33 GB 与 12.59→4.97 GB）；
- 单项示例（Table 3）：GMAN 在 PEMS07 上 GPU cost 16.22→4.71 GB（GMAN-SGL），MAE 17.00→16.20、RMSE 28.64→28.16；[[staeformer|STAEformer]] 在 PEMS07 上 GPU cost 22.11→9.34 GB、MAE 19.22→19.16（PEMS04 上 MAE 18.25→18.29）。

运行效率方面，论文报告并行框架的 SGL 变体利用并行性显著加速（GMAN 在 PEMS07 上训练时间 166.77→80.48 s/epoch，Table 3）；串行模型依赖 RNN 迭代特征提取、无法并行，其 SGL 变体运行效率与原型相当，节点规模大时效率更优（Sec. 4.1）[^src-lets-group]。Table 3 中也有反例方向的数据：DGCRN 在 PEMS03 上训练时间 106.95→125.32 s/epoch，即串行模型在小规模数据集上可能更慢。

论文另比较了 STWave 的两种注意力变体：STWave-ESGAT 也能降低内存，但依赖预定义矩阵确定注意力范围；STWave-SGL 无此限制，论文据此称其适用面更宽（Sec. 4.1，论文自述）[^src-lets-group]。

### 消融（Q2）

子图划分方式对比（Table 4，DGCNet(R) 上）：随机等分（-RD）、DTW 相似度矩阵 + METIS 划分（-METIS）、映射矩阵聚类子图表示（-DC）均劣于 SGPM——PEMS04 MAE 18.49（SGL）vs 18.71（RD）/ 18.68（METIS）/ 18.90（DC）；PEMS08 MAE 13.66 vs 14.07 / 13.91 / 14.34。论文归因：RD 不考虑节点相关性故最差；METIS 的子图固定、不随节点相关性动态变化；DC 把节点特征压缩为子图表示、丢失单节点特征（Sec. 4.2）[^src-lets-group]。

聚合方式对比（Table 5）：平均优于 max 与 sum——PEMS04 MAE 18.49（平均）vs 18.60（max）/ 18.71（sum）；PEMS08 MAE 13.66 vs 13.81 / 13.83，其中 PEMS08 上 sum 的 MAPE（9.00%）反而略低于平均（9.01%），是表中唯一的反向格，论文正文称「平均结果最佳」、未讨论该例外（Sec. 4.2）。论文归因：sum 造成特征冗余；max 只保留最大特征、忽略其余子图的信息（Sec. 4.2）[^src-lets-group]。

### 超参数（Q3）

在 PEMS07 + DDGCRN-SGL 上做 K、M 敏感性实验（Fig. 4）：K 或 M 过小时性能骤降，论文归因于子图数或每子图节点数不足、时空特征提取不充分导致欠拟合；当 K×M 接近或超过 N 时，性能与原模型相当且内存显著更少。论文据此指出 K、M 的配置是平衡内存效率与性能的关键（Sec. 4.3）[^src-lets-group]。

### 子图可视化（Q4）

对 DDGCRN-SGL 在测试集上生成的子图做 T-SNE 降维可视化（Fig. 5）：子图内节点呈现聚类行为，不同子图之间存在共享或位置相邻的节点。论文用这一观察佐证节点跨子图重叠确实存在、SGFAM 的平均聚合有必要（Sec. 4.4）[^src-lets-group]。

## 适用范围与论文自述边界

- 实验限于交通网络数据集（PEMS03/04/07/08）与交通预测任务（Sec. 4），其他时空数据类型未在论文中验证[^src-lets-group]。
- 内存收益随规模增大而增大：论文自述「随数据集规模增大，SGL 的内存节省更显著」（Sec. 4.1，PEMS08 18.2% vs PEMS07 60.5% 的对照）[^src-lets-group]。
- 串行（RNN 型）backbone 无法从子图并行中获益，论文报告其 SGL 变体运行效率与原型相当、仅在大节点规模时更优（Sec. 4.1）[^src-lets-group]。
- K、M 按数据集逐一配置（Table 2），配置不当会导致欠拟合（Sec. 4.3）[^src-lets-group]。

## 与其他方法的关系

- 定位：SGL 不提出新的时空预测架构，而是替换现有 STGNN 空间特征提取中「全图 O(N²) 相关矩阵」这一环节，使相关矩阵从 N×N 变为 K×K[^src-lets-group]。这与压缩优化器状态/激活的系统级内存技术（见 [[memory-efficient-training]]）作用在不同层面（本课程层面的对照，非论文原文表述）。
- 与 [[patchstg|PatchSTG]] 的关系：PatchSTG 用 leaf KDTree 按地理坐标做不规则空间分块（KDD 2025）[^src-patchstg]；本文在相关工作中将其与 FCGCN（Louvain）、LarSTL（METIS）一并归为静态划分，并提出按特征相似度动态划分的 SGPM 作为对照方案（Sec. 1–2，论文自述）[^src-lets-group]。
- 与 [[node-visibility|VisiFold 的子图采样]] 的关系：VisiFold 把剩余节点随机划分为固定大小子图以提升并行度、兼作隐式正则化；SGL 按记忆向量相似度划分，并用跨子图平均聚合显式处理节点重叠[^src-visifold][^src-lets-group]。
- 相关工作节还提及 MoE 路线（论文称 EXPERT，引 Lee and Ko 2024，即 [[testam|TESTAM]]）与预训练路线的 STEP、[[std-mae|STD-MAE]]，把它们归为「提升精度但计算开销大」的一类（Sec. 2，论文自述）[^src-lets-group]。

## 被引情况与口径差异

FENCE（AAAI-26）的 Related Work 在 Spatial-Temporal Imputation 段将本文（Weng et al. 2025）与 BRITS（Cao et al. 2018）、GRU-D（Che et al. 2018）并列归入「判别式插补模型」[^src-fence]。但本文原文的任务设定是时空图预测（Sec. 3.1），实验全部为交通预测的 MAE/RMSE/MAPE 评测（Sec. 4, Table 3），未包含插补实验。两类口径以本文原文为准；该引用语境差异已同步记录在 [[source-fence]] 与 [[fence]]。

## 另见

- [[source-lets-group]] — 论文摘要页
- [[large-scale-spatial-temporal-graph]] — 大规模时空图预测的问题域与方法分类
- [[traffic-forecasting]] — 交通预测任务总览
- [[memory-efficient-training]] — 系统级内存优化技术对比

[^src-lets-group]: [[source-lets-group]]
[^src-visifold]: [[source-visifold]]
[^src-fence]: [[source-fence]]
[^src-patchstg]: [[source-patchstg]]
