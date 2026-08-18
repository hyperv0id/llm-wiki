---
title: "Stationarity-Aware Retrieval"
type: concept
tags:
  - retrieval-augmented
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

# Stationarity-Aware Retrieval

平稳性感知检索（Stationarity-Aware Retrieval）是一种检索增强预测范式，核心思想是：相似度检索的可靠性依赖数据集的平稳性，因此检索策略（候选选择和证据融合）应根据数据集的平稳性水平自适应调整[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

## 核心动机

传统检索增强预测隐含假设"相似的过去意味着相似的未来"，但这一假设在非平稳数据集上不成立。在平稳数据集上（如 Electricity，ADF 平稳比 97.2%），输入相似度排名与未来相似度排名的 Spearman ρ=1.000；在非平稳数据集上（如 Exchange，ADF 平稳比 12.5%），ρ 降至 0.285[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。这意味着纯相似度检索在非平稳条件下变得脆弱。

## 核心组件

平稳性感知检索包含三个自适应机制[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]：

1. **时间对齐增强**（[[time-aligned-retrieval-enhancement]]）：在形态相似度之上叠加时间对齐奖励，提供额外的匹配维度。
2. **多样性控制**（[[diversity-based-retrieval-selection]]）：通过平稳性调节的 MMR 平衡系数 λ(s̄)，低平稳性 → 更强多样化 → 覆盖异质 regime。
3. **聚合锐度调节**：Gaussian 核带宽 σ(s̄)，低平稳性 → 更大 σ → 更平滑的权重分配 → 更鲁棒。

## 与传统检索增强的区别

| 维度 | 纯相似度检索 | 平稳性感知检索 |
|------|-------------|-------------|
| 候选选择 | Top-K 相似度 | 时间对齐 + 多样性 MMR |
| 融合策略 | 固定权重 | 平稳性条件化 Gaussian 带宽 |
| 冗余处理 | 无 | MMR 去冗余 |
| 非平稳适应 | 降权（被动） | 多样化 + 平滑聚合（主动） |

传统方法在检索不可靠时通过降权（down-weighting）抑制检索分支，但这有两个局限：(i) 未解决输入检索与未来相关性的根本失配；(ii) 过度抑制部分有用邻居，使模型退化为纯参数预测器[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。平稳性感知检索则主动改善检索质量本身。

## 应用

[[saraf|SARAF]]（KDD 2026）是首个系统实现平稳性感知检索的框架，在 8 个数据集上验证有效，尤其对非平稳数据集（如 Exchange）增益显著[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

## 相关概念

- [[dataset-stationarity-estimation]] — 平稳性估计方法
- [[time-aligned-retrieval-enhancement]] — 时间对齐检索增强
- [[diversity-based-retrieval-selection]] — 多样性检索选择
- [[retrieval-augmented-spatio-temporal-forecasting]] — 检索增强时空预测
- [[gtr]] · [[pir]] · [[ratd]] · [[source-raf]] — 其他检索增强预测方法
- [[nsdiff]] — 非平稳扩散（从概率角度处理非平稳性）

[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]: [[source-stationarity-aware-retrieval-augmented-forecasting-kdd26]]
