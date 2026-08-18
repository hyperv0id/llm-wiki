---
title: "SARAF: Stationarity-Aware Retrieval-Augmented Time Series Forecasting"
type: source-summary
tags:
  - retrieval-augmented
  - time-series-forecasting
  - non-stationarity
  - diversity-based-retrieval
  - stationarity
  - kdd-2026
created: 2026-08-19
last_updated: 2026-08-19
source_count: 1
confidence: medium
status: active
---

# SARAF: Stationarity-Aware Retrieval-Augmented Time Series Forecasting

**Authors**: Shiqiao Zhou (U Birmingham), Holger Schöner (Siemens AG), Zipeng Wu (U Birmingham), Edouard Fouché (Siemens AG), IAG Wilson (U Birmingham), Shuo Wang (U Birmingham)

**Venue**: KDD 2026 | **Code**: [github.com/ShiqiaoZhou/SARAF](https://github.com/ShiqiaoZhou/SARAF)

## 核心论点

论文首先通过诊断实验揭示：相似度检索的可靠性强依赖于数据集的平稳性（stationarity）。在平稳数据集 Electricity 上，输入相似度排名与未来相似度排名的 Spearman ρ=1.000；而在非平稳数据集 Exchange 上 ρ 仅为 0.285，意味着"相似的过去"未必带来"相似的未来"[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。此外，相似度检索在滑动窗口数据库上的 Top-K 结果经常高度冗余，在非平稳条件下会将不匹配的案例聚合为误导性"共识"[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

## 贡献

1. **实证发现**：相似度检索的可靠性随平稳性变化——非平稳数据集上输入-未来相似度排名一致性显著下降[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。
2. **SARAF 框架**：结合 (i) 时间对齐检索增强和 (ii) 平稳性控制的多样性检索与自适应融合，由数据集级平稳性估计器指导[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。
3. **实验验证**：在 8 个真实数据集上达到竞争性性能，相比 RAFT 平均 MSE 降低 3.85%、MAE 降低 1.87%；相比 DUET 平均 MSE 降低 4.05%、MAE 降低 0.75%[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

## 方法概要

SARAF 流程：(1) 对输入窗口归一化后，计算与 TS 数据库中所有历史窗口的 Pearson 相关相似度，叠加时间对齐奖励（hour-of-day, day-of-week, month-of-year 等）；(2) 取 Top-M 候选池，通过平稳性控制的随机 MMR（Maximal Marginal Relevance）选择 K 个多样化候选；(3) 以平稳性条件化的 Gaussian 核加权聚合检索到的未来段；(4) 与直接线性预测平均融合后线性投影输出[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

平稳性估计：将每个窗口分为 W 个子窗口，计算子窗口均值和标准差的跨窗口变异，经全局尺度归一化后取平均得到数据集级平稳性分数 s̄ ∈ [0,1][^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。低平稳性 → 较小的 λ（更强多样化）、较大的 σ（更平滑的聚合）；高平稳性 → 较大的 λ（更依赖相似度）、较小的 σ（更尖锐的聚合）。

## 局限（论文自述）

1. 当前使用全局相似度函数，更细粒度的通道级或组级相似度可能更好捕获异质动态，但会增加检索和存储开销[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。
2. 检索数据库由密集滑动窗口构建，对长序列和大数据集内存密集；压缩数据库是未来方向[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。
3. 平稳性控制在数据集级别估计，可能无法充分反映实例级 regime shift[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

## 相关链接

- [[saraf]] — 框架实体页
- [[stationarity-aware-retrieval]] — 平稳性感知检索概念
- [[time-aligned-retrieval-enhancement]] — 时间对齐检索增强技术
- [[diversity-based-retrieval-selection]] — 多样性检索选择技术
- [[dataset-stationarity-estimation]] — 数据集级平稳性估计
- [[gtr]] · [[pir]] · [[ratd]] · [[source-raf]] — 其他检索增强预测方法
- [[nsdiff]] · [[retrieval-augmented-spatio-temporal-forecasting]] — 非平稳与检索增强相关

[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]: [[source-stationarity-aware-retrieval-augmented-forecasting-kdd26]]
