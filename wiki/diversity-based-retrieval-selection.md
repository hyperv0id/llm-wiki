---
title: "Diversity-Based Retrieval Selection"
type: technique
tags:
  - retrieval-augmented
  - diversity
  - mmr
  - non-stationarity
  - time-series-forecasting
created: 2026-08-19
last_updated: 2026-08-19
source_count: 1
confidence: medium
status: active
---

# Diversity-Based Retrieval Selection

多样性检索选择（Diversity-Based Retrieval Selection）是一种在检索增强预测中通过平稳性控制的随机 MMR（Maximal Marginal Relevance）从候选池中选择多样化子集的技术，旨在减少冗余并覆盖异质历史 regime[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

## 动机

在滑动窗口数据库中，相似度检索的 Top-K 结果常高度冗余——近重复的证据浪费有限的检索预算。在非平稳条件下，冗余但不匹配的案例会聚合为误导性"共识"，放大误差[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

## 机制

### 平稳性控制的平衡参数

原始 MMR 使用平衡系数 λ ∈ [0,1] 在相关性（λ=1）和多样性（λ=0）间权衡。多样性检索选择将 λ 设为数据集平稳性分数 s̄ 的函数[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]：

$$\lambda(\bar{s}) = \lambda_{min} + \bar{s}(\lambda_{max} - \lambda_{min})$$

低平稳性数据集（较小 s̄）→ 较小 λ → 更强多样化；高平稳性数据集 → 较大 λ → 更依赖相似度[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

### 随机 MMR

从 Top-M 候选池中选择 K 个（M≫K）[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]：

1. 首先选最相似候选作为锚点。
2. 迭代选择剩余 K−1 个：对每个候选 x_i 计算 MMR(i) = λ(s̄) · S_sim(q,i) − (1−λ(s̄)) · max_{x_j∈R} Δ(i,j)。
3. 冗余代理 Δ(i,j) = 1 − |S_sim(q,i) − S_sim(q,j)|，避免 O(M²) 的全配对相似度计算。
4. 从 softmax 分布中采样下一候选（而非确定性 argmax），增加随机性。

### 复杂度

标准 MMR 的候选间冗余计算复杂度为 O(M²d)，此方法使用查询相似度代理后降至 O(MK)[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

## 实验证据

在 [[saraf|SARAF]] 的消融中[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]：
- 移除多样性在非平稳数据集 Exchange 上导致明显性能下降，证明自适应多样性控制对覆盖异质 regime 重要[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。
- SARAF 平均降低检索集内相似度 14.57%（相比去除多样性的变体）和 16.09%（相比 RAFT），有效缓解冗余[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。
- 在 ETTh2 上，多样性使检索融合未来与真实未来之间的相似度提升最高达 34.0%（H=192）[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

## 相关概念

- [[stationarity-aware-retrieval]] — 平稳性感知检索概念
- [[dataset-stationarity-estimation]] — 平稳性估计驱动 λ 自适应
- [[saraf]] — 实现该技术的框架
- [[retrieval-augmented-spatio-temporal-forecasting]] — 检索增强预测范式

[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]: [[source-stationarity-aware-retrieval-augmented-forecasting-kdd26]]
