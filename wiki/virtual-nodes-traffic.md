---
title: "Virtual Nodes for Long-term Traffic Prediction"
type: technique
tags:
  - traffic-forecasting
  - graph-neural-network
  - graph-rewiring
  - long-term-forecasting
  - adaptive-adjacency
created: 2026-08-30
last_updated: 2026-08-30
source_count: 2
confidence: medium
status: active
---

# Virtual Nodes for Long-term Traffic Prediction

Virtual Nodes（虚拟节点）是在图上追加的、与全部真实节点相连的辅助节点，使信息能在单个 GNN 层内完成全图聚合。Zhuang 等（MIT，arXiv:2501.10048，2025-01）将这一源自分子图表示学习的图改写（graph rewiring）手段引入长期交通预测：与直接改边的 graph rewiring 不同，虚拟节点不改动原图拓扑，只在邻接矩阵中追加行/列，充当从网络各处汇聚信息再向全网广播的枢纽[^src-virtual-nodes]。论文自述虚拟节点在交通预测中的应用此前基本未被探索[^src-virtual-nodes]。

## 问题：长期预测与 over-squashing

论文引用的分类标准将 1 小时以内的预测归为短期（Wang et al., 2020），超过 1 小时视为中长期（Hou et al., 2014；Yu et al., 2018）；此前研究主要将 ST-GNN 用于短期预测（5 分钟粒度下通常不超过 60 分钟 horizon）[^src-virtual-nodes]。其机制解释是：消息传递逐层只聚合邻居信息，l 跳外的信息至少需要 l 层才能到达目标节点，而随层数堆叠，远处节点的信息被压缩进固定尺寸的向量表示——即论文沿用的 over-squashing 概念（Alon & Yahav, 2020）[^src-virtual-nodes]。论文以事故影响沿路网传播为例说明这一延迟会妨碍 ST-GNN 及时捕捉突发事件对远端路况的影响（Sec 2.3, Fig 1）[^src-virtual-nodes]。

over-squashing 的另一类缓解路线是用 global attention 让信息一步到达任意节点：论文指出 Graph Transformer 类工作以全局注意力处理全局信息，但其二次方空间与内存开销限制在更大路网上的实用性，因此相关技术转向改写图结构的路线（Sec 3.1）[^src-virtual-nodes]。[[graphgps|GraphGPS]] 的全局注意力分支是这一路线的对照例子（wiki 侧对照，非论文所举工作）。

## 机制：半自适应邻接矩阵

论文以 [[stgcn|STGCN]]（GCN 空间 + TCN 时间）为基座模型 fθ（Sec 4.1），在其邻接矩阵中集成 nv 个虚拟节点，并提出两种连接权重的来源（Sec 4.2）[^src-virtual-nodes]：

- **Adaptive（全任务驱动）**：为全部节点（含虚拟节点）配置两组可学习嵌入 E1、E2 ∈ R^{(|V|+nv)×d}，计算反对称矩阵 Aadapt = ReLU(E1·E2ᵀ − E2·E1ᵀ)。论文引用 Wu et al.（2020）的结论——时序预测中学到的关系应为单向——作为采用该反对称形式的依据；随后以阈值 r 剪除弱连接（式 6-7）[^src-virtual-nodes]。该反对称构造与 [[mtgnn|MTGNN]] 图学习层（[[graph-learning-layer]]）的公式同族但更简：MTGNN 版为 ReLU(tanh(α(M₁M₂ᵀ−M₂M₁ᵀ))) 并配 top-k 稀疏化[^src-mtgnn]，论文版不含 tanh(α·) 投影与 top-k（改用上文阈值 r 剪枝）；论文未明言沿用 MTGNN 公式。
- **Semi-adaptive（半自适应）**：将距离邻接矩阵 Adist 与 Aadapt 中对应虚拟节点的分块拼接为分块矩阵 Asemi——左上为真实节点间的 Adist，右上/左下为虚拟-真实连接，右下为虚拟节点间连接（式 8-9）。地理先验由此保留，任务驱动的权重只负责虚拟节点接入[^src-virtual-nodes]。

虚拟节点的初始信号置零，从零开始经训练学习聚合真实节点信息（Sec 4.2）[^src-virtual-nodes]。论文称该方法可移植到任何接受邻接矩阵输入的 ST-GNN 或图模型（Sec 5.3 自述）[^src-virtual-nodes]。

## 论文报告的实验

**设置**（Sec 5.1-5.3）：LargeST 基准的 San Diego（SD）子集——716 传感器、17,319 条边、平均度 24.2、密度 0.0338、5 分钟采样、2017-01-01 至 2021-12-31 共 525,888 帧（Table 1）；训练/测试使用 2019-2020 共 35,040 帧（论文原文如此；按 5 分钟采样，2019-2020 两年应为 210,528 帧、一年应为 105,120 帧，均与 35,040 不吻合，论文未解释该数字）；预测步长 1-20 个 horizon（每个 5 分钟，即 5-100 分钟），指标为 RMSE 与 MAPE。作者报告选择 LargeST 中最小的 SD 子集是为了在计算需求与模型参数复杂度间取得平衡，并提及在信息传播本已高效的小图上叠加虚拟节点可能引入过拟合（Sec 5.1）[^src-virtual-nodes]。

**主结果**（Table 2）：在距离基线、"All-ones"（单虚拟节点全 1 连接）、Adaptive（1/2/5/10/20 个虚拟节点）、Semi-adaptive（1/2/5/10/20 个虚拟节点）共 12 种配置中，作者报告 Semi-10 V.N.（10 个虚拟节点）在全部 4 个 horizon 与 2 个平均列上的 RMSE 与 MAPE 均为最低。在 75-100 分钟平均区间，Semi-10 为 RMSE 42.32、MAPE 0.1735，对比距离基线 45.15、0.1827，论文自述对应 RMSE 降低约 6.27%、MAPE 降低约 5.04%（Sec 5.3）[^src-virtual-nodes]。All-ones 在平均与长程区间略优于距离基线，但部分 horizon 上反而更差（Sec 5.3）[^src-virtual-nodes]。图 5 显示半自适应配置的 RMSE/MAPE 优势随预测步长增大而扩大（Fig 5）[^src-virtual-nodes]。

**敏感性**（Sec 5.4, Fig 6）：Adaptive 配置在全部 horizon 上均不优于距离基线（Sec 5.4 自述；注：Table 2 中 Adaptive-20 在 H15/H20 两格 MAPE 略低于距离基线，与该自述存在出入），作者认为这可能是额外参数带来的过拟合与不稳定（原文为 may imply 的推测语气）；Semi-adaptive 随虚拟节点数从 1 增至 10 性能改善、在 10 处最优、增至 20 时回落（Fig 6b）[^src-virtual-nodes]。

**可解释性**（Sec 5.5, Fig 7）：将学到的真实-虚拟邻接权重映射回路网热图后，作者报告虚拟节点 3、8、10 与大量真实节点存在显著更强的连接（其中虚拟节点 8 最强），且虚拟节点 8 的高权重真实节点多位于交叉口与关键枢纽处，即权重自动集中到交通活跃区域[^src-virtual-nodes]。

## 范围与局限

- 实验仅覆盖 SD 一个子数据集（LargeST 中最小），更大路网上的可扩展性未验证；论文将时变邻接矩阵与更大规模验证列为未来工作（Sec 6 自述）[^src-virtual-nodes]。
- 基座模型仅 STGCN 一种；论文称未来将纳入更多模型（Sec 5.3 自述）[^src-virtual-nodes]。
- 实验的预测步长为 5-100 分钟；按论文引用的分类（Wang et al., 2020；Hou et al., 2014；Yu et al., 2018），超过 1 小时为中长期，报告的长程改进区间为 75-100 分钟平均（Table 2）[^src-virtual-nodes]。
- 长期增益依赖虚拟节点数量这一超参数（最优值 10 出现在该实验设置中，Sec 5.4 自述）[^src-virtual-nodes]。

## 与 wiki 中相关设计点的关系

- [[gwnet|GWNet]] 与 [[mtgnn|MTGNN]] 确立了交通预测中自适应邻接矩阵范式；论文的 Aadapt 采用反对称单向构造并以 MTGNN（Wu et al., 2020）的单向性结论为依据，扩展到虚拟节点接入，而 semi-adaptive 的分块设计重新引入被纯自适应方法舍弃的距离先验（Sec 3.2 论文定位）[^src-virtual-nodes]。
- [[stgformer|STGformer]] 代表另一条避免逐层堆叠的长程建模路线（单层线性时空注意力，详见该页）；论文方案则保留多层消息传递、以虚拟节点缩短信息路径[^src-virtual-nodes]。
- 辨析：over-squashing（长程信息压缩，本文动机）与 over-smoothing（层间表征趋同）是不同的表征退化机制，后者见 [[over-smoothing-in-gnns|Over-smoothing]]，两者的专门区分另见 [[over-squashing]]。

## 相关页面

- [[source-virtual-nodes]] — 源文件摘要
- [[traffic-forecasting]] — 交通预测方法总览
- [[large-scale-spatial-temporal-graph]] — LargeST 基准与大规模路网预测
- [[stgcn]] — 基座模型
- [[gwnet]] — 自适应邻接矩阵范式（交通预测）
- [[mtgnn]] — 论文所引单向性结论的出处（Wu et al., 2020）
- [[graph-learning-layer]] — MTGNN 图学习层（同族公式）
- [[over-squashing]] — 本文动机问题
- [[graphgps]] — over-squashing 的 global attention 缓解路线对照
- [[over-smoothing-in-gnns]] — 相邻的 GNN 表征退化病理
- [[stgformer]] — 单层长程建模的另一路线对照

[^src-virtual-nodes]: [[source-virtual-nodes]]
[^src-mtgnn]: [[source-mtgnn]]
