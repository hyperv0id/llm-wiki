---
title: "GraphGPS：通用、强大、可扩展的图 Transformer 配方"
type: technique
tags:
  - graph-transformer
  - graph-neural-network
  - positional-encoding
  - linear-attention
  - message-passing
  - neurips-2022
created: 2026-08-30
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# GraphGPS：通用、强大、可扩展的图 Transformer 配方

GraphGPS 是 Rampášek 等人（Mila / NTU / Valence Discovery）提出的模块化图 Transformer（Graph Transformer, GT）框架[^src-graphgps]。论文把构建 GT 组织为一条三要素配方（recipe，3 main ingredients）：（i）位置/结构编码（positional/structural encoding, PE/SE）的嵌入模块，（ii）local message-passing 机制（MPNN），（iii）global attention 机制，三者组合出一个 GPS 层，并按此蓝图实现为建立在 PyG 与 GraphGym 之上的模块化软件包 GraphGPS[^src-graphgps]。

## 问题：GT 的两个缺口

论文指出的第一个缺口是设计基础：GT 论文众多，但「什么构成好的位置或结构编码、它们之间有何区别」缺乏共同基础，因此论文先给出 PE/SE 的定义与分类（Abstract, Sec 3.1）[^src-graphgps]。第二个缺口是规模：全连接 GT 的全局注意力需要 $O(N^2)$ 计算量，把已有 GT 限制在几百节点的图上；论文摘要自述 GPS 是首个节点与边复杂度线性（$O(N+E)$）的图 Transformer 架构（「first」为论文自述归因），做法是把 local 真实边聚合与全连接 Transformer 解耦（Abstract, Sec 1）[^src-graphgps]。

## GPS 层：MPNN 与全局注意力逐层并行

GPS 层的更新方程（Sec 3.3, Eq. 1–4）：

$$X^{l+1}, E^{l+1} = \mathrm{GPS}^l(X^l, E^l, A)$$
$$X_M^{l+1}, E^{l+1} = \mathrm{MPNN}_e^l(X^l, E^l, A), \qquad X_T^{l+1} = \mathrm{GlobalAttn}^l(X^l)$$
$$X^{l+1} = \mathrm{MLP}^l(X_M^{l+1} + X_T^{l+1})$$

要点：MPNN 分支接收节点与边特征，全局注意力分支只接收节点特征，边特征不进入注意力分支；两支输出逐元素相加后经一个 2 层 MLP，residual 连接与 batch normalization 在公式中省略（Sec 3.3）[^src-graphgps]。MPNN 与 GlobalAttn 都是可替换模块——MPNN 可以是任何作用于局部邻域的函数（论文实例化 GINE、GatedGCN、PNA），GlobalAttn 可以是任何全连接层（论文实例化 Transformer、Performer、BigBird）（Sec 3.3, Fig 1）[^src-graphgps]。

为什么不在 Transformer 前面先堆 MPNN 层（如 GraphTrans 的两段式）？论文的论证是：MPNN 受 [[over-smoothing-in-gnns|over-smoothing]]、[[over-squashing]] 和 1-WL 表达力限制，前置的 MPNN 层可能在早期就不可逆地丢失信息；GraphTrans 被论文相关工作部分描述为首个 hybrid 架构，但 GPS 选择在每一层内并行交错两条分支，而非分两段（Sec 2, Sec 3.3）[^src-graphgps]。

## PE/SE 分类

论文把编码组织为 local / global / relative 三类、与 PE/SE 两个维度交叉（Sec 3.1, Table 1, Fig 1）[^src-graphgps]：PE 提供距离概念（两个节点在图或子图中越近，其 PE 越接近），SE 提供结构相似性概念（两个节点的 m-hop 子图越相似，其 local SE 越接近），二者不同且互补（Sec 3.1）[^src-graphgps]。代表性条目：local PE 为随机游走矩阵非对角元素按行求和（Fig 1 原文；Table 1 同一条目写作按列求和，论文图与表在行/列表述上不一致）；global PE 为 Laplacian/邻接矩阵特征向量（可经 SignNet、DeepSet 处理符号与数量歧义）；relative PE 为节点对最短路径/热核/随机游走距离或特征向量梯度；local SE 为节点度、m 步随机游走矩阵对角线（即 [[rwse|RWSE]]）、子结构计数、Ricci 曲率；global SE 为特征值或图的直径、连通分量数等整体属性；relative SE 为节点对结构差异的边特征（Table 1）[^src-graphgps]。论文的实证评估聚焦 global PE、relative PE 与 local SE 三种（Sec 3.1）[^src-graphgps]。

论文用两类图论证 MPNN 需要这些编码（Sec 3.2）[^src-graphgps]：标准 MPNN 的表达力与 1-WL 测试相当；在 CSL（Circular Skip Link）图对上 MPNN 与 1-WL 全部失败，而 Laplacian PE 赋予节点可区分的初始颜色、随机游走对角线 local SE 能捕捉 skip-link 差异；在 Decalin 分子图上，1-WL/MPNN 与 local SE 都不能区分 (a,d) 与 (b,d) 两个候选链接，需要 relative PE 或 global PE[^src-graphgps]。

## 复杂度与表达力

复杂度方面，论文给出的口径是 GPS 的计算复杂度对节点数与边数均为线性 $O(N+E)$，该口径不含某些 PE 可能需要的预计算步骤（如 Laplacian 特征分解）（Sec 3.3）[^src-graphgps]。实现路径：把 PE/SE 限制在真实节点与边的特征上、把边特征排除在注意力之外，即可避免实例化 $N \times N$ 注意力矩阵，从而使用 $O(N)$ 的线性注意力（Performer、BigBird），MPNN 分支为 $O(E)$；对分子图、正则图、知识图谱等稀疏图 $E = \Theta(N)$，整体可视为 $O(N)$（Sec 3.3）[^src-graphgps]。作者报告即使在小型分子图上也有实测差距：约 6M 参数的模型在 ogbg-molpcba 上每 epoch 需 196s，而 SAN 为 883s（同型号 GPU）（Sec 3.3）[^src-graphgps]。

表达力方面，论文论证两条（Sec 3.4）[^src-graphgps]：其一，边特征经 MPNN 层隐式写入节点表示，使注意力分支虽只用节点特征也能间接利用边信息，逐层交错降低了 GraphTrans/SAT 式一次性消息传递的表示瓶颈；其二，沿用 SAN 的论证，在给出全集 Laplacian 特征向量时，该架构是图上的通用函数逼近器，参数足够时可强于任何 WL 同构测试（Sec 3.4, 附录 C）[^src-graphgps]。

## 实验证据（作者报告口径）

消融实验覆盖 4 个数据集（ZINC、PCQM4Mv2 子集、CIFAR10、MalNet-Tiny），种子数按数据集而异：ZINC 与 CIFAR10 每项 4 个随机种子，PCQM4Mv2 子集与 MalNet-Tiny 用 3 个（Table 2, 附录 A.2/附录 B）[^src-graphgps]：

- 去掉 MPNN 分支导致所有数据集性能大幅下降（ZINC MAE 0.070 → 0.217，PCQM4Mv2 子集 0.1159 → 0.3294），且无 MPNN 时网络会过拟合 PE/SE（Table 2a, Sec 4.1）[^src-graphgps]。
- 全局注意力对除 ZINC 外的数据集均有益；作者解释 ZINC 的目标（cLogP 与 SA-score）都是局部子结构计数，不需要全局连接但需要结构编码（Table 2a, Sec 4.1）[^src-graphgps]。Performer 的预测性能低于全秩 Transformer 但优于无注意力基线并能扩展到大图；BigBird 在其实验设置中无显著增益且更慢（Table 2a, Sec 4.1）[^src-graphgps]。
- PE/SE 消融（Table 2b）：Table 2b 图注称 RWSE 增益一致且计算代价相对低，SignNet+DeepSets 是该消融中单项最佳编码但计算代价更高；Sec 4.1 进一步报告 RWSE 对分子数据更有益、LapPE 对图像超像素更有益（详见 [[rwse|RWSE]] 页）[^src-graphgps]。

基准结果（Tables 3–6）[^src-graphgps]：Benchmarking-GNNs 五项上，ZINC MAE 0.070±0.004（论文称该表 SOTA）、MNIST 98.051、CIFAR10 72.298、PATTERN 86.685、CLUSTER 78.016（Table 3）[^src-graphgps]；OGB 四项图级任务上 molhiv AUROC 0.7880（作者注明该数据集出现过拟合但仍高于 SAN）、molpcba 0.2907、ppa 0.8015、code2 0.1894，GPS 在 molpcba/ppa/code2 均位列前三，且在四项上都优于其他对比 GT（code2 上 SAT 除外）（Table 4, Sec 4.2）[^src-graphgps]；PCQM4Mv2 上 GPS-medium 验证 MAE 0.0858（19.4M 参数），以不到一半的参数量优于 GRPE（0.0890, 46.2M）、EGT（0.0869, 89.3M）、Graphormer（0.0864, 48.3M），训练集过拟合更小，且 RWSE 只依赖图结构、无需 Graphormer 式的 3D 构象距离预计算（Table 5 及其图注, Sec 4.2）[^src-graphgps]；MalNet-Tiny（最高 5,000 节点的函数调用图）上 Performer 版 92.72%±0.7、Transformer 版 93.36%±0.6，高于原基准最佳 GIN 的 90%（Sec 4.2）[^src-graphgps]；LRGB 五项中四项 GPS 优于全部对比基线，PascalVOC-SP F1 0.3748、COCO-SP 0.3412、Peptides-func AP 0.6535、Peptides-struct MAE 0.2500、PCQM-Contact MRR 0.3337（Table 6, Sec 4.2）[^src-graphgps]。作者总结：在 16 个任务中 11 项超过所有对比图 Transformer、8 项取得当时最优（Sec 5，论文自述口径）[^src-graphgps]。

## 论文自述的局限

结论部分作者列出两条局限（Sec 5）[^src-graphgps]：图 Transformer 对超参数敏感，不存在对所有数据集通用的单一配置；缺少迫使模型利用长程依赖的困难图数据集，使线性注意力架构的可扩展优势难以充分展现[^src-graphgps]。论文另以假说形式承认：local MPNN 分支的必要性可能与图数据量有限有关，Graphormer 在完整 PCQM4Mv2 上不用 local 模块也表现很好（Sec 4.1）[^src-graphgps]。

## wiki 侧谱系定位（非论文内容）

本 wiki 把 GraphGPS 作为 hybrid 局部/全局架构的参照框架：每层内 MPNN 分支与注意力分支并行求和的设计，与 [[topology-aware-graph-transformer|TGT]] 在注意力矩阵内做凸组合、[[over-smoothing-in-gnns|over-smoothing]] 条目中「全局注意力绕过迭代平滑」的机制归类相互对照；[[wire|WIRE]] 的图基准实验即在 GraphGPS + ReLU Performer 骨干上进行（见 [[source-2509-22259]]）。

## 相关页面

- [[rwse]] — 论文使用并消融的随机游走结构编码（local SE）
- [[topology-aware-graph-transformer]] — 注意力矩阵层面的 local/global 混合（对照设计）
- [[over-smoothing-in-gnns]] — GPS 引用其作为 MPNN 动机的核心问题
- [[quest-attention]] — 被列为潜在应用场景的注意力变体
- [[wire]] — 后续在 GraphGPS 骨干上评估的图位置编码

[^src-graphgps]: [[source-graphgps]]
