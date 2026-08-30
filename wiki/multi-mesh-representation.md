---
title: "Multi-Mesh Representation"
type: technique
tags:
  - graph-neural-network
  - multi-resolution
  - spherical-geometry
  - weather-forecasting
  - message-passing
created: 2026-08-30
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# Multi-Mesh Representation

**Multi-mesh（多分辨率网格表征）** 是 [[graphcast|GraphCast]] 引入的内部图表征：由正二十面体（icosahedron）逐级细分得到的多级球面网格，叠加全部层级的边后形成一张空间近似均匀、同时含局部短边与长程长边的消息传递图（Figure 1g）[^src-graphcast]。

## 动机：为什么不用经纬网格

论文给出两条理由：

1. **空间不均匀**：经纬网格的格点分布空间不均匀，极区分辨率过高，要求不成比例的算力投入；multi-mesh 在全球范围内空间分辨率近似均匀（Supp Sec 3.2）[^src-graphcast]。
2. **交互范围**：CNN 限于局部 patch（或规则扩张的较长程）；Transformer 虽可全局交互，但面对 GraphCast 输入中百万级以上的格点受二次方内存复杂度限制；GNN 可建模任意稀疏交互模式（Supp Sec 3.2）[^src-graphcast]。

## 构造

- **细分**：从单位半径正二十面体（M0：12 节点、20 个三角面）出发，每次细分将每个三角面一分为四、新节点重投影回球面；迭代 6 次得 M6：40,962 节点、81,920 面（Supp Sec 3.3、Table 4）[^src-graphcast]。细分-重投影使三角边长存在差异：论文脚注注明最大差异 16.4%、标准差 6.5%（Supp Sec 3.3 脚注 4）[^src-graphcast]。
- **多级边叠加**：M(r-1) 的节点是 Mr 节点的子集，因此各低层级网格的边可直接叠加到最细网格上；低层级引入的节点成为长程通信的 hub。multi-mesh = M6 的全部节点 + M0–M6 所有层级边的并集，构成扁平的长短边混合层级（Supp Sec 3.3）[^src-graphcast]。
- **规模**：按双向边计（Table 4 口径），M6 自身边 245,760 条，叠加 M0–M6 后 multi-mesh 共 327,660 条 mesh 边；另有 Grid2Mesh 有向边 1,618,746 条（格点与 mesh 节点距离 ≤ 0.6 × M6 最长边即连接，保证每个格点至少连一个 mesh 节点）、Mesh2Grid 有向边 3,114,720 条（每个格点连到包含它的三角面的 3 个顶点）（Table 4、Supp Sec 3.3）[^src-graphcast]。

处理器（16 层 GNN）只在 multi-mesh 上运行，多级边使消息传递在少数步内即可完成长程信息传播（Supp Sec 3.2/3.5）[^src-graphcast]。

## 消融证据

论文在补充材料中训练了去掉多级边的消融模型：架构不变（编码器/解码器与节点数相同），处理器只在 M6 自身边（245,760 条边）上做消息传递，因而只能以短边传播信息（Supp Sec 7.3.1）[^src-graphcast]。作者报告：完整 multi-mesh 在除 50 hPa 层 5 天以上时效外的全部预测变量上优于消融模型，位势（全部气压层）与 5 天内海平面气压改进最明显；与 HRES 的 scorecard 对照显示，multi-mesh 是 GraphCast 在位势 5 天内时效上超越 HRES 的必要结构（Figure 29）[^src-graphcast]。

## 适用范围

- 该设计服务于球面上近似均匀、格点数量巨大（约 10^6 量级）的场演化建模；编码器/解码器不要求规则矩形网格，论文注明可作用于任意 mesh 型状态离散化（Supp Sec 3.2）[^src-graphcast]。
- 消融结论限于 GraphCast 的训练设置、变量集与 2018 年验证协议（论文补充材料实验）[^src-graphcast]。

## 相关页面

- [[graphcast]] — 使用 multi-mesh 的主模型
- [[spherical-geometry-inductive-bias]] — 球面几何归纳偏置视角下的路线对照
- [[circular-patching]] — CirT 的替代球面处理方案（按纬线分块）
- [[storminsight]] — 复用「GraphCast 风格」multi-mesh 消息传递作为大气环境编码器

[^src-graphcast]: [[source-graphcast]]
