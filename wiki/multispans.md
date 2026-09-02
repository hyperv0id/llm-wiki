---
title: "MultiSPANS"
type: technique
tags:
  - traffic-forecasting
  - transformer
  - structural-entropy
  - spatio-temporal
  - attention-mask
created: 2026-09-02
last_updated: 2026-09-02
source_count: 1
confidence: medium
status: active
---

# MultiSPANS

MultiSPANS（Multi-range Spatiotemporal Prediction Attention Network with Structural entropy optimization）是论文提出的交通状态预测时空 Transformer（WSDM 2024），作者来自北航、中南大学、清华与滴滴，代码开源于 SELGroup/MultiSPANS[^src-multispans]。

## 论文针对的问题

论文指出时空 Transformer 的两个问题[^src-multispans]：

1. 处理长时间序列时效果不如预期——离散时间点上的 token 信息不足以学习成对注意力并建模高阶全局时间性；
2. 难以直接利用图结构——主流做法要么融合 GNN 与 Transformer 的输出，要么从网络上取简单的注意力掩码/编码，这些结构学习机制没有理论指导，忽略了丰富的结构信息。

设计动机来自城市功能分区的层次性：同一高层社区内的道路流量特征相近，分区反映在路网上、难以预定义，因此希望无监督、自适应地从路网中导出层次结构[^src-multispans]。

## 机制

整体结构：MFCL 模块 → N 个（实验取 3）交错排列的时间/空间 Transformer 编码器（残差连接，各编码器 skip 求和）→ 输出层（转置 1D 卷积 + MLP）[^src-multispans]。

**多滤波卷积模块（MFCL）**。受视觉 Transformer patching 启发，用两组滤波扩展 token 通道并注入局部时空模式[^src-multispans]：

- 时间卷积滤波：m 个不同尺寸的 1D 滤波提取多频率短程时间模式，输出沿通道拼接（c → c_t）；基本实现取 4 个尺寸 1×1、1×2、1×3、1×6 的滤波（约对应 5/10/15/30 分钟间隔），序列两端复制首末点补齐到 T + k_j − 1。滤波尺寸/数量按任务定制，时间滤波的 stride 可调大以压缩序列——这是长历史窗口低成本注入的机制。
- 图卷积滤波：对归一化邻接 Â = D⁻¹(A+I) 取 h 次幂做多跳聚合，各跳输出拼接为 d = (h+1)·c_t 维；论文称比同类多跳设计（引 MixHop 等）可训练参数更少。

**位置嵌入层**。空间用归一化拉普拉斯矩阵 I − D^(−1/2)AD^(−1/2) 的 k 个最小非平凡特征向量线性投影（引 Dwivedi & Bresson）；时间用正弦位置编码；另对 day-of-week 与 hour-of-day 做 one-hot 映射为跨批次周期嵌入 D_b。时间/空间 Transformer 分别输入 H + D_t + D_b 与 H + D_s + D_b[^src-multispans]。

**时间/空间 Transformer**。统一的多头注意力模块，注意力 logits 为 QK 内积加加性相似度矩阵 S/√d，再与掩码 M 逐元素相乘。两个 Transformer 的差别：位置编码不同（D_t vs D_s）；时间 Transformer 在时间点之间建模、所有空间位置共享一组投影参数（空间侧相反）；S 与 M 只为空间注意力设计[^src-multispans]。

**多尺度图结构感知（Multi-Range Graph Structure Perception）**，三步[^src-multispans]：

1. 路网抽象：借鉴 deDoc 的 combination/merge 树算子，从平铺编码树出发每轮贪心执行使结构熵下降最多的节点对与算子，熵不再下降时终止，得到最优编码树 T*；
2. 多层注意力掩码：编码树每层对应图节点集的一个划分（特定空间尺度的潜在分区）；第 l 层掩码 M^(l) 把同层同一社区的节点对置 1、否则置 −INF。L 层树给出 L−1 个非叶层掩码，另加邻接矩阵作第 L 个掩码以覆盖最小范围的边级局部关系；L 个掩码分给 H 个注意力头中的 L 个（要求 H > L），其余 H−L 个头不掩码、保持全局注意力；
3. 层级相关分数：定义相对结构熵以刻画树节点间的相对复杂度与信息量；两个叶节点的分数经由最低公共祖先 θ 的路径求和得到（式 10），等价视角是把编码树视为图、累加两叶间最短有向路径上相连节点的相对结构熵。得到的层级相关矩阵 S_hier 作为先验分数加到注意力矩阵上，充当相对位置编码。

**输出层**。各 ST 编码器与 MFCL 的中间输出经 skip 求和后，若隐藏长度 T 与预测长度 T′ 不一致，由转置一维卷积平滑扩展序列，再由 MLP 投影到目标形状[^src-multispans]。

## 证据

实验设置：单卡 RTX 3090，Adam + MAE 损失，50 epoch，lr 1e-2，batch 32；统一配置 k=3 层、d=64、h=8 头；数据按 6:2:2 划分；基线实现来自 LibCity[^src-multispans]。

- 主实验（PEMSD4/8 × flow/speed 共 4 子集，12 个基线）：论文报告相对 SOTA 平均提升 MAE 2.57%、MAPE 2.16%、RMSE 3.78%；最强项 PEMSD8-speed 为 MAE 1.36 / MAPE 2.84 / RMSE 3.26（+4.23%/+3.73%/+4.96%）；PEMSD4-flow MAE 19.07 / RMSE 30.46，PEMSD8-flow RMSE 23.87；论文把 RMSE 优势归因于 MFCL 与转置卷积输出层的平滑去噪作用[^src-multispans]。
- 长窗口实验（表 2，PEMSD4-flow）：stride 1/3/4 把 12/36/48 步历史压成统一 12 长度隐藏态。48 步窗口、8 个时间滤波（尺寸 1,2,3,4,6,12,18,24）时 MAE 18.85 / MAPE 13.17 / RMSE 30.18，参数 332.3K、269.15s/epoch；同窗 STTN MAE 19.31、699.8K 参数、931.18s/epoch（论文称其时间开销与收益不成比例），STGCN MAE 20.97、参数随窗口从 385.9K 涨到 1565.5K[^src-multispans]。
- 消融（表 3，PEMSD4-flow，数字为去掉组件后的 MAE 恶化幅度）：去 MFCL −5.24%（时间滤波 1.98% > 空间滤波 1.49%）；去多层掩码 −2.33%；去层级相关分数 −1.52%；两者都去 −4.55%；掩码换成 Infomap 层次社区检测只剩 −1.83%——论文以此论证结构熵最小化比 Infomap 更适合路网层次抽象[^src-multispans]。
- 可视化：多层掩码让不同注意力头分别捕获不同粒度的空间依赖；无掩码的全局注意力大多依赖少数关键节点、邻接矩阵掩码则丢失复杂语义；基于多层注意力选出的与节点 197 最相关的 top-3 节点，其流量曲线比 vanilla 注意力选出的整体更相似[^src-multispans]。
- 超参：时间滤波数 k1 与空间滤波跳数 k2 的影响稳定；即使 k1=1 或 k2=0，RMSE 仍为 30.98 / 30.92，高于多数基线[^src-multispans]。

## 范围

实验仅覆盖 PEMSD4/8 两个数据集，论文未报告更大规模路网或跨数据集迁移；编码树由给定路网图一次性导出，论文未讨论掩码与分数随时间的动态更新。论文自述未来工作是把结构熵引导的注意力机制推广到图与空间数据，并从层次网络分析视角分析 Transformer 的可解释性[^src-multispans]。

## 在 wiki 中的位置

- 空间注意力掩码路线上接 [[pdformer|PDFormer]]（地理/语义二值掩码）；MultiSPANS 用结构熵导出的多层掩码与相对结构熵位置编码替代手工设计，见 [[structural-entropy]][^src-multispans]。
- [[source-stg-mamba|STG-Mamba]]（arXiv 2024）后续把 MultiSPANS 列为对比基线并报告在三项指标上一致超越[^src-stg-mamba]。

## 相关页面

[[traffic-forecasting]] · [[pdformer]] · [[structural-entropy]] · [[graph-node-clustering]] · [[stgcn]] · [[gwnet]] · [[mtgnn]] · [[dcrnn]] · [[source-astgcn|ASTGCN]]

[^src-multispans]: [[source-multispans]]
[^src-stg-mamba]: [[source-stg-mamba]]
