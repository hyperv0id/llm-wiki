---
title: "Source: Let's Group — A Plug-and-Play SubGraph Learning Method for Memory-Efficient Spatio-Temporal Graph Modeling"
type: source-summary
tags:
  - spatiotemporal
  - traffic-forecasting
  - subgraph-learning
  - memory-efficiency
  - ijcai-2025
created: 2026-08-29
last_updated: 2026-08-29
source_count: 2
confidence: medium
status: active
---

# Source: Let's Group (IJCAI-25)

**作者**: Wenchao Weng, Hanyu Jiang, Mei Wu, Xiao Han, Haidong Gao, Guojiang Shen, Xiangjie Kong（通讯；浙江工业大学、杭州电子科技大学 ITMO 联合学院）
**发表**: IJCAI-25 官方 proceedings, pp. 3471–3479。raw 文件 `raw/weng-lets-group-ijcai-2025.pdf` 已核实为 IJCAI-25 official proceedings 排版：每页页眉 "Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)"、页码 3471–3479、PDF 元数据 Subject 为 "Paper accepted and presented at IJCAI-2025"；用户著录的论文集编号 No. 386 未在 PDF 正文与元数据中出现。被引来源：FENCE（AAAI-26）参考文献（著录页码与 PDF 一致）。
**代码**: github.com/wengwenchao123/SubGraphLearning

## 核心论点

论文提出即插即用子图学习方法 SGL，处理 STGNN 空间特征提取的内存瓶颈：图卷积/注意力的相关矩阵为 O(N²)，内存随节点数增长（Sec. 3.6）[^src-lets-group]。SGPM 用 M 个随机初始化的可学习记忆向量作锚点，按 softmax 相似度取 top-K 节点构成可重叠子图（Eq. 5–7）；各子图共享同一 G() 构造 K×K 相关矩阵（Eq. 8–9）；SGFAM 将同一节点跨子图的特征取平均以消除重叠冗余（Eq. 10–12）[^src-lets-group]。总体复杂度 O(NM+MK+K²+N)，论文称 M 为常数时简化为 O(N+K²)（Sec. 3.6）[^src-lets-group]。

## 关键结果（作者报告）

- 四个 PEMS 交通数据集（170–883 节点，Table 1）、8 个模型行（GMAN、STWave 两种注意力变体、[[staeformer|STAEformer]]、DGCRN、DDGCRN、DGCNet 两种变体）上，SGL 变体预测性能与原模型相当，平均 GPU 内存开销最高降 56.4%（摘要，未说明平均口径）；DDGCRN 在 PEMS08/PEMS07 分别降 18.2%/60.5%（Sec. 4.1, Table 3）[^src-lets-group]。
- 消融（Table 4–5）：SGPM 优于随机划分、DTW+METIS 静态划分与聚类表示；平均聚合优于 max/sum[^src-lets-group]。
- 超参（Fig. 4）：K×M 接近或超过 N 时性能与原模型相当；K/M 过小欠拟合（Sec. 4.3）[^src-lets-group]。

## 局限与边界

- 实验仅覆盖交通数据集与预测任务；串行 RNN 型 backbone 不产生并行加速，其 SGL 变体运行效率与原型相当、大节点规模时更优（Sec. 4.1）[^src-lets-group]。
- 口径提示：FENCE 将本文归入判别式时空插补模型，但本文无插补实验，任务设定为交通预测（Sec. 3.1、Sec. 4）[^src-lets-group][^src-fence]。

[^src-lets-group]: [[source-lets-group]]
[^src-fence]: [[source-fence]]
