---
title: "Dataset-Level Stationarity Estimation"
type: concept
tags:
  - stationarity
  - non-stationarity
  - time-series-forecasting
  - adaptive-retrieval
created: 2026-08-19
last_updated: 2026-08-19
source_count: 1
confidence: medium
status: active
---

# Dataset-Level Stationarity Estimation

数据集级平稳性估计（Dataset-Level Stationarity Estimation）是一种基于矩统计量的高效平稳性度量方法，通过计算滑动窗口数据库中各段局部统计量的跨窗口变异来估计数据集的平稳性水平[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

## 动机

经典平稳性检验（如 ADF [Augmented Dickey-Fuller] 和 KPSS）是单变量的，在大规模多变量数据库上计算代价高（ADF 需对每个通道的完整序列做滞后回归拟合，复杂度 O(CT·p_max³)），不适合作为检索管道中的实时控制信号[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

## 方法

对于样本 x_i ∈ R^{L×C}，将其分为 W 个不重叠子窗口[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]：

1. 对每个子窗口计算逐通道局部均值和标准差。
2. 计算子窗口均值和标准差的跨窗口变异 v_μ 和 v_σ（即子窗口统计量的标准差）。
3. 以全局尺度项 σ̄（数据集 {x_i} 的标准差）归一化，使分数尺度不变：

$$\tilde{s}_i = \frac{1}{2}\left[\left(1 - \min\left(1, \frac{v_\mu}{\bar{\sigma}}\right)\right) + \left(1 - \min\left(1, \frac{v_\sigma}{\bar{\sigma}}\right)\right)\right] \in [0,1]$$

4. 数据集级平稳性为所有样本的平均：s̄ = (1/N) Σ s̃_i[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

s̄ 越大表示局部均值和方差随时间越稳定（更平稳）[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

## 复杂度

每窗口复杂度 O(LC + WC)，总计 O(NLC)，即 O(Nd)。可高效批量 GPU 计算[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

## 与 ADF 的对比

SARAF 同时报告了 ADF 平稳比（各通道 ADF 检验拒绝单位根的通道比例）和提出的平稳性分数 s̄。两者呈正相关趋势：ADF 平稳比低的数据集 s̄ 也较小，ADF 平稳比高的数据集 s̄ 较大[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。但 s̄ 直接可在滑动窗口数据库上计算，无需独立回归拟合，适合检索控制。

## 在 SARAF 中的作用

s̄ 驱动两个自适应机制[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]：
- MMR 平衡系数 λ(s̄)：低 s̄ → 更强多样化
- Gaussian 核带宽 σ(s̄)：低 s̄ → 更平滑聚合

## 各数据集平稳性分数

| 数据集 | s̄ | ADF 平稳比 |
|--------|------|-----------|
| ETTh1 | 0.7041 | 100.0% |
| ETTh2 | 0.5731 | 57.1% |
| ETTm1 | 0.6466 | 100.0% |
| ETTm2 | 0.6080 | 85.7% |
| Exchange | 0.4203 | 12.5% |
| Solar | 0.7439 | 100.0% |
| Electricity | 0.8648 | 97.2% |
| Traffic | 0.8628 | 100.0% |

来源：SARAF 论文 Table 1[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

## 相关概念

- [[stationarity-aware-retrieval]] — 平稳性感知检索
- [[diversity-based-retrieval-selection]] — 多样性选择由 s̄ 驱动
- [[saraf]] — 使用该方法的框架
- [[nsdiff]] — 另一种处理非平稳性的方法（概率扩散视角）

[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]: [[source-stationarity-aware-retrieval-augmented-forecasting-kdd26]]
