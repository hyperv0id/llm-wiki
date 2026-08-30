---
title: "RWSE：随机游走结构编码"
type: technique
tags:
  - graph-neural-network
  - structural-encoding
  - positional-encoding
  - random-walk
  - graph-transformer
created: 2026-08-30
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# RWSE：随机游走结构编码（Random-Walk Structural Encoding）

RWSE（random-walk structural encoding）是 GraphGPS 论文使用的名称，指「m 步随机游走矩阵对角线作为节点特征」的编码，论文将其归类为 local SE（局部结构编码）（Sec 3.1, Table 1, Fig 1, Sec 4.1）[^src-graphgps]；论文并未自称首次命名该编码。Table 1 将该编码的出处归因于 Dwivedi 等人的可学习结构/位置表示工作（LSPE，其参考文献 [16]）[^src-graphgps]。本 wiki 此前对 RWSE 无任何记载，本页为其首个条目。

## 机制

对图的随机游走矩阵取 $m$ 步幂，其对角线元素逐节点拼成特征向量；GraphGPS 的 Figure 1 与 Table 1 将其描述为「m 步随机游走矩阵的对角线」并作为节点特征使用（Fig 1, Table 1）[^src-graphgps]。编辑注（非论文原话）：$(D^{-1}A)^m$ 的对角线在随机游走语义下对应「从该节点出发、$m$ 步后返回该节点」的游走量，这是随机游走矩阵的标准性质。

论文给出的解释性例子：对奇数 $m$，该对角线可以指示一个节点是否属于长度为 $m$ 的环（Sec 3.1）[^src-graphgps]。local SE 的语义定义是：半径 $m$ 内两个节点的 m-hop 子图越相似，其 local SE 越接近（Table 1）[^src-graphgps]。

## 表达力证据与边界

GraphGPS 用 1-WL（1-Weisfeiler-Leman）视角论证这类编码的价值：标准 MPNN 的表达力与 1-WL 相当，在 CSL（Circular Skip Link）图对上 MPNN 与 1-WL 全部失败，而随机游走对角线形式的 local SE 能捕捉两个图在 skip-link 上的差异、产生不同的节点着色（Sec 3.2；GraphGPS 将该论证归因于其参考文献 [42, 16]）[^src-graphgps]。

边界同样来自论文（Sec 3.2）：在 Decalin 分子图上，节点 a 与 b、c 与 d 在 1-WL/MPNN 着色下各自同色，local SE 也无法区分 (a,d) 与 (b,d) 两个候选链接；该案例需要基于距离的 relative PE 或基于特征向量的 global PE 才能区分（Sec 3.2）[^src-graphgps]。

## 消融证据（GraphGPS 作者报告，Table 2b；种子数按数据集而异：ZINC 与 CIFAR10 每项 4 个随机种子，PCQM4Mv2 子集与 MalNet-Tiny 3 个，附录 A.2/B）

| 数据集（指标） | 无 PE/SE | +RWSE |
|---|---|---|
| ZINC（MAE ↓） | 0.113 | 0.070 |
| PCQM4Mv2 子集（MAE ↓） | 0.1355 | 0.1159 |
| CIFAR10（Acc ↑） | 71.49 | 71.96 |
| MalNet-Tiny（Acc ↑） | 92.64 | 92.77 |

Table 2b 图注称 RWSE「以相对较低的计算代价提供一致增益」，并称 SignNet+DeepSets 是该消融中单项最佳编码但计算代价更高（Table 2b 图注）[^src-graphgps]。作者进一步报告编码收益与数据类型相关：RWSE 对分子数据更有益，Laplacian 特征向量编码（LapPE）对图像超像素更有益（Sec 4.1）[^src-graphgps]。

## 使用方式（以 GraphGPS 实验为准）

- 维度选择：ZINC 使用 RWSE-20（20 步）配 linear 编码器（Table A.2）；ogbg-molhiv 与 ogbg-molpcba 使用 RWSE-16（Table A.3）；PCQM4Mv2 子集上 GPS-small 的消融使用 RWSE-16（Table B.2）[^src-graphgps]。
- 与 3D 信息的对照：PCQM4Mv2 实验中作者强调，Graphormer 需要从近似 3D 分子构象预计算空间距离，而其使用的 RWSE 只依赖图结构（Sec 4.2）[^src-graphgps]。
- 作为基线组件：LRGB 基准的 SAN+RWSE 与 GatedGCN+RWSE 变体出现在 GraphGPS 的对比表中（Table 6）[^src-graphgps]。

## 复杂度

GraphGPS 论文未给出 RWSE 的渐近复杂度记号；Table 2b 图注的「相对较低的计算代价」是论文的定性表述，附录以各数据集的墙钟时间（含 PE precompute 列）报告开销，而非复杂度分析（Table 2b 图注, 附录 A.3）[^src-graphgps]。

## 相关页面

- [[graphgps]] — 命名、归类并消融 RWSE 的框架
- [[over-smoothing-in-gnns]] — MPNN 表达力受限的另一动机
- [[wire]] — WIRE 论文另有一个避免谱分解、稀疏图上 $O(N)$ 计算的 RWPE 变体，名称相近；机制见 [[wire]] 与 [[source-2509-22259]]

[^src-graphgps]: [[source-graphgps]]
