---
title: "Spatial Tokenizer (Adjacency-Matrix Patches)"
type: technique
tags:
  - traffic-forecasting
  - tokenization
  - graph-structure
  - explicit-graph-modeling
  - generalization
  - vit
created: 2026-07-27
last_updated: 2026-07-27
source_count: 1
confidence: medium
status: active
---

# Spatial Tokenizer (Adjacency-Matrix Patches)

**邻接矩阵 patch 空间 tokenizer** 是 [[stunet|STUNet]]（KDD 2026）的显式空间编码模块：把关系图邻接 $A\in\mathbb{R}^{N\times N}$ 切成固定大小非重叠 patch，再经 MLP 映为统一维度的 spatial tokens，使可变 $N$ 的路网结构进入与时序同维度的 token 空间，且**不依赖传感器观测值**[^src-stunet]。

## 动机

点级编码整张邻接复杂度灾难；node-wise embedding 又与具体传感器绑定、换网失效。ViT 式 patch 在**结构矩阵**上对齐维度、保留局部连接模式（立交、环岛、Y 形分流等可迁移组件），把空间变成可预训练、可冻结的“基”[^src-stunet]。

## 做法

1. 将 $A$ 划分为 patch；若 $N$ 不能被 stride 整除则 zero-pad 边界。
2. 每个 patch 经 MLP 映到 $d$ 维 → $E_s$。
3. **Stage 1**：autoencoder 预训练；输入邻接的**节点索引随机置换**，同一传感器集对应多种矩阵实现，逼 tokenizer 学置换稳健的结构模式。
4. **Stage 2**：参数**冻结**接入 [[query-aggregate-attention|Query-Aggregate Attention]]，避免时间损失回写空间表示——论文认为这是跨网络泛化的关键[^src-stunet]。

附录：对直路 / Y 形 / 环采样 spatial token，UMAP 聚类 ARI 0.96、NMI 0.98，说明 token 能区分原型拓扑[^src-stunet]。

## 与相近 tokenization

| 技术 | 切什么 | 目的 |
|------|--------|------|
| 本技术（STUNet） | 邻接矩阵块 | 显式结构、跨网零样本 |
| [[irregular-spatial-patching\|Irregular Spatial Patching]]（[[patchstg\|PatchSTG]]） | 传感器经纬度点（leaf KDTree） | 大规模动态空间 attention 降复杂度 |
| [[independent-spatial-tokenization\|Independent Spatial Tokenization]]（SIFusion） | 栅格场（海冰 SIC） | 解耦空间编码与时间序列 |
| [[spatial-temporal-tokenizer\|Spatial-Temporal Tokenizer]]（STD-PLM） | 节点动态+内禀特征 → PLM token | 服务预训练语言模型骨干 |

## Related

- [[stunet]] · [[source-stunet]] · [[query-aggregate-attention]]
- [[patchstg]] · [[ood-generalization]]

[^src-stunet]: [[source-stunet]]
