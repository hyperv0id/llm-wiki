---
title: "Recipe for a General, Powerful, Scalable Graph Transformer (GraphGPS)"
type: source-summary
tags:
  - graph-transformer
  - graph-neural-network
  - positional-encoding
  - linear-attention
  - neurips-2022
created: 2026-08-30
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# Recipe for a General, Powerful, Scalable Graph Transformer (GraphGPS)

**Authors**: Ladislav Rampášek（Mila, Université de Montréal）、Mikhail Galkin（Mila, McGill）、Vijay Prakash Dwivedi（NTU Singapore）、Guy Wolf（Mila）、Anh Tuan Luu（NTU）、Dominique Beaini（Valence Discovery, Mila）
**Venue**: NeurIPS 2022（第 36 届会议标识见 PDF 第 1 页，PDF 内核实）；对应 raw 文件 `raw/graphgps-rampasek-2022.pdf`（arXiv:2205.12454v4, 2023-01-15）

## 核心论点

论文提出构建图 Transformer 的三要素配方（recipe，3 main ingredients）：PE/SE 嵌入模块、local message-passing（MPNN）模块、global attention 模块；单个 GPS 层内 MPNN 分支（接收边特征）与全局注意力分支（不接收边特征）并行，输出相加后过 2 层 MLP（Sec 3.3, Eq. 1–4）[^src-graphgps]。论文摘要自述这是首个节点与边复杂度 $O(N+E)$ 的图 Transformer 架构，做法是把 local 真实边聚合与全连接注意力解耦，使 Performer/BigBird 等线性注意力进入图域（Abstract, Sec 1, Sec 3.3）[^src-graphgps]。论文同时给出 PE/SE 的 local/global/relative 分类（Sec 3.1, Table 1），并以 CSL 图与 Decalin 分子论证 MPNN 受 1-WL 表达力限制、需要编码补充（Sec 3.2）[^src-graphgps]。

## 证据（作者报告）

消融（Table 2；ZINC 与 CIFAR10 用 4 seeds，PCQM4Mv2 子集与 MalNet-Tiny 用 3 seeds，附录 A.2/B）：去掉 MPNN 分支性能大幅下降（ZINC MAE 0.070→0.217）；注意力消融中 Transformer 对除 ZINC 外数据集均有益，Performer 次之、可扩展，BigBird 无显著增益；Table 2b 图注称 RWSE 增益一致且计算代价相对低，SignNet+DeepSets 为单项最佳编码[^src-graphgps]。基准（Tables 3–6）：ZINC MAE 0.070±0.004；PCQM4Mv2 验证 MAE 0.0858（19.4M 参数，少于 GRPE/EGT/Graphormer 一半）；MalNet-Tiny（最高 5,000 节点）Transformer 版 93.36%；LRGB 五项中四项优于全部对比基线；作者总结 16 任务中 11 项超过所有对比 GT、8 项当时最优（Sec 5 自述）[^src-graphgps]。

## 局限（论文自述，Sec 5）

图 Transformer 对超参数敏感，无 one-size-fits-all 配置；缺少需要长程依赖的图数据集，线性注意力的可扩展优势难以充分体现[^src-graphgps]。作者另以假说形式指出 MPNN 分支的必要性可能与图数据量有限有关（Sec 4.1）[^src-graphgps]。

## 相关页面

- [[graphgps]] — 技术页
- [[rwse]] — 论文使用并消融的结构编码

[^src-graphgps]: [[source-graphgps]]
