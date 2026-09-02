---
title: "PDFormer"
type: technique
tags:
  - traffic-forecasting
  - transformer
  - spatio-temporal
  - attention
created: 2026-08-30
last_updated: 2026-09-02
source_count: 7
confidence: medium
status: active
---

# PDFormer

PDFormer（Propagation Delay-aware dynamic long-range transFormer）是论文提出的面向交通流预测的时空 Transformer 模型（AAAI 2023），作者为 Jiawei Jiang 与 Chengkai Han（共同一作，北航）、Wayne Xin Zhao（中国人民大学高瓴人工智能学院）与 [[jingyuan-wang|Jingyuan Wang]]（通讯作者，北航），代码开源于 BUAABIGSCITY/PDFormer[^src-pdformer-jiang-2023]。该实验室的其他工作还包括 [[bigcity|BIGCity]]、GTG 与 [[hifinet|HiFiNet]][^src-craft]。

## 论文针对的问题

论文认为当时基于 GNN 的交通预测模型有三条局限[^src-pdformer-jiang-2023]：

1. 空间依赖按静态建模（预定义图或自学习图固定），而真实城市中节点间相关性随时间变化，如早晚高峰时增强、其余时段减弱；
2. 局部化设计只能覆盖短程空间信息，GNN 的 over-smoothing 使长程空间依赖难以捕获——而城市功能分区使得两个相距很远的节点可能呈现相似的交通模式；
3. 消息传递是即时的（immediate message passing），忽略了交通状况在位置之间传播存在时间延迟，如事故的影响需要几分钟才扩散到邻区。

## 机制

整体结构为数据嵌入层 → L 层时空编码器 → 输出层[^src-pdformer-jiang-2023]。

**数据嵌入。** 除全连接输入嵌入与时间位置编码外，用归一化拉普拉斯矩阵 $\Delta = I - D^{-1/2}AD^{-1/2}$ 的 $k$ 个最小非平凡特征向量做线性投影，得到空间图拉普拉斯嵌入；论文引 Dwivedi et al. 2020 说明拉普拉斯特征向量把图嵌入欧氏空间并保留全局结构。另叠加周索引（1–7）与日索引（1–1440）两个周期嵌入[^src-pdformer-jiang-2023]。

**空间自注意力（SSA）与两个图掩码。** 在每个时间片对节点做 self-attention，注意力分数逐片计算，因此空间依赖是动态的。纯 SSA 等价于全连接图，论文引入两个二值掩码矩阵筛出关键节点对[^src-pdformer-jiang-2023]：

- 地理掩码 $M^{geo}$：图上 hop 距离小于阈值 $\lambda$ 的节点对置 1，覆盖短程依赖（GeoSSA）；
- 语义掩码 $M^{sem}$：用 DTW 度量历史流量序列相似度，每个节点取 top-K 为语义邻居，覆盖"距离远但城市功能相似"的长程依赖（SemSSA）。

**延迟感知特征变换（DFT）。** 用大小为 $S$ 的滑动窗口切历史序列，经 k-Shape 聚类得到 $N_p$ 个短时交通模式质心；相似度权重由节点近 $S$ 步历史的嵌入与模式记忆向量的内积 softmax 得到，据此对模式质心加权求和得到整合表示 $r_{t,n}$，拼接后用于更新地理空间注意力的 key（$\tilde{K}_t = K_t + R_t$）。论文只把 DFT 接在 GeoSSA 上，理由是远处节点的短期流量对当前节点影响小[^src-pdformer-jiang-2023]。

**时间自注意力（TSA）与异构头融合。** 每个节点在时间维做 self-attention 捕获动态时间模式；三类注意力头（GeoSAH、SemSAH、TAH）拼接后经 $W^O$ 投影，$d' = d/(h_{geo}+h_{sem}+h_t)$，其后是 position-wise FFN、LayerNorm 与残差[^src-pdformer-jiang-2023]。

**输出层。** 每层后接 1×1 卷积 skip 连接并求和，直接一次性输出多步预测；论文说明选直接式而非递归式是考虑累积误差与效率[^src-pdformer-jiang-2023]。

## 证据

论文在 6 个公开数据集上实验：图数据 PeMS04/07/08（307/883/170 节点，过去 12 步预测未来 12 步），网格数据 NYCTaxi、CHIBike、T-Drive（6 步预测 1 步，含 inflow/outflow）；17 个基线分四类，含 [[stgcn|STGCN]]、[[gwnet|GWNet]]、[[mtgnn|MTGNN]]、STSGCN、STGODE、STGNCDE 与注意力系的 STTN、GMAN、TFormer、ASTGNN[^src-pdformer-jiang-2023]。

- 论文报告在全部数据集、全部指标上优于所有基线（Student's t-test，0.01 水平），相对第二名平均提升 MAE 4.58%、MAPE 5.00%、RMSE 4.79%；表 2 中 PeMS04 MAE 18.321（次优 ASTGNN 18.601）、PeMS07 MAE 19.832（次优 20.616）、PeMS08 MAE 13.583（次优 14.974）[^src-pdformer-jiang-2023]。
- 效率（表 4，PeMS04 每 epoch）：训练 133.871s、推理 8.120s，对比 ASTGNN 的 208.724s / 52.016s，论文称训练与推理时间分别降低 35% 与 80% 以上，并归因于去掉了 GMAN、ASTGNN 的 encoder-decoder 结构[^src-pdformer-jiang-2023]。
- 消融（PeMS04、NYCTaxi inflow）：SSA 换成 GCN、去掉两个掩码、去掉 GeoSAH 或 SemSAH、去掉 DFT，性能都下降——论文由此论证动态长程注意力、双掩码与延迟建模各自的贡献[^src-pdformer-jiang-2023]。
- 注意力可视化（case study）：无掩码时注意力弥散或集中于高流量干道；加掩码后聚焦邻近区域与远处功能相似区域，论文以北京区域 592 关注国贸立交、二环附近功能相近区域为例[^src-pdformer-jiang-2023]。

实验设置：单卡 RTX 3090 + 128GB 内存，AdamW（lr 0.001）、batch 16、训练 200 epoch，隐藏维 $d \in \{16,32,64,128\}$ 与层数 $L \in \{2,4,6,8\}$ 按验证集搜索，全部实验重复 10 次取平均[^src-pdformer-jiang-2023]。

## 在后续工作与 wiki 中的位置

- PDFormer（2023）属于基于注意力的全局时空依赖建模路线[^src-pdformer-jiang-2023]；[[hyperd|HyperD]] 论文把它与 STTN、GMAN、[[staeformer|STAEformer]] 并列归入该路线，并在 PeMS03/04/07/08 上报告超越它，长时域优势更明显（[[traffic-forecasting]] 按此归类型整理路线）[^src-hyperd-hybrid-periodicity-decoupling]。
- [[multispans|MultiSPANS]]（WSDM 2024）在 related work 中把 PDFormer 概括为"在注意力头上采用地理与语义空间掩码"的代表性工作[^src-multispans]；其多层掩码可视为这条掩码路线的推广——用结构熵最小化从路网无监督导出编码树，逐层社区划分生成多粒度掩码，并以相对结构熵作加性位置编码（见 [[structural-entropy]]）。
- [[st-unification]] 与 [[metadg|MetaDG]] 的分类把它归入 dynamics-aware / dynamic topology 一档：动态性体现在空间注意力，但元参数等模型中间量仍是静态的[^src-metadg]。
- 它是后续轻量化与效率对比的常用参照：后续论文报告其参数量约 1,404K、推理计算量约 1.771 GFLOPs[^src-lightweight-mixed-graph-unrolling]，训练显存约 8295MB[^src-hephestus]。注意这些数字来自后续论文的测量表格，PDFormer 原文只报告运行时间（表 4），未报告参数量与显存。

## 范围

论文的实验范围是 PeMS 高速公路与城市网格客流/出行数据上的流量预测；DFT 的 k-Shape 模式库、DTW 语义邻居都依赖历史数据，掩码阈值 $\lambda$、语义邻居数 $K$、窗口 $S$、聚类数 $N_p$ 为超参。论文自述的未来工作包括推广到风电预测等其他时空任务，以及引入预训练解决数据不足[^src-pdformer-jiang-2023]。

## 相关页面

[[traffic-forecasting]] · [[st-unification]] · [[metadg]] · [[hyperd]] · [[hephestus]] · [[staeformer]] · [[std-mae]] · [[opencity]] · [[stgcn]] · [[gwnet]] · [[mtgnn]] · [[dcrnn]] · [[graph-learning-as-self-attention]] · [[algorithm-unrolling]] · [[multispans]] · [[structural-entropy]]

[^src-pdformer-jiang-2023]: [[source-pdformer-jiang-2023]]
[^src-craft]: [[source-craft]]
[^src-hyperd-hybrid-periodicity-decoupling]: [[source-hyperd-hybrid-periodicity-decoupling]]
[^src-metadg]: [[source-metadg]]
[^src-lightweight-mixed-graph-unrolling]: [[source-lightweight-mixed-graph-unrolling]]
[^src-hephestus]: [[source-hephestus]]
[^src-multispans]: [[source-multispans]]
