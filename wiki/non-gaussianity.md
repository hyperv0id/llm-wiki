---
title: "Non-Gaussianity (Time-Series Window Complexity)"
type: concept
tags:
  - time-series
  - probabilistic-forecasting
  - evaluation
  - data-characterization
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: medium
status: active
---

# Non-Gaussianity（时序窗口非高斯性）

**Non-Gaussianity** 在 [[probts|ProbTS]] 中用作**预测窗口内数据分布复杂度**的代理指标：衡量该窗口内观测值分布偏离高斯的程度[^src-probts]。它与趋势强度 $F_T$、季节强度 $F_S$ 一起，构成解释点/概率方法场景偏好的三类数据特征[^src-probts]。

## 动机

点预测方法以 MSE 等损失优化时，等价于在高斯输出头 + MAP 假设下的概率建模，因此内生地假设窗口值近似高斯[^src-probts]。高级概率方法（流、扩散等）不预设闭式分布，可数据驱动地拟合更复杂律[^src-probts]。故“非高斯性”可区分：何时简单点头足够，何时需要灵活分布估计[^src-probts]。

## 量化方式

对长度等于预测 horizon 的时序片段：

1. 拟合该窗口观测值的高斯分布；
2. 用 **Jensen–Shannon 散度**度量经验值分布与该高斯的差异；
3. 对数据集内所有片段取平均，得到数据集级非高斯性[^src-probts]。

实现上短程窗口常用长度 30，长程常用 336[^src-probts]。**值越大表示越偏离高斯、分布越复杂**[^src-probts]。

## 与方法表现的关系（ProbTS）

- 短程场景中 Solar-S 等**高非高斯性**数据集上，概率模型相对定制点预测架构的 CRPS/NMAE 优势更明显[^src-probts]。
- 长程场景许多数据集非高斯性偏低、趋势/季节更突出，定制 NAR 点架构更占优[^src-probts]。
- 基础模型侧：MOIRAI（混合预定义分布头）与 Chronos 等在高非高斯性上相对 [[csdi|CSDI]] 的 CRPS 落差更大，说明闭式/混合头对复杂分布表达不足[^src-probts]。

## 相关特征对照

| 指标 | 含义 | 高值时常见方法偏好（经验） |
|------|------|---------------------------|
| 趋势 $F_T$ | STL 趋势相对残差强度 | 长程 AR 更易误差累积；RevIN 常有帮助[^src-probts] |
| 季节 $F_S$ | STL 季节相对残差强度 | AR 解码可占优（如 Traffic）[^src-probts] |
| 非高斯性 | JS(经验 ‖ 高斯) | 需要灵活概率建模；点头/简单分布头吃亏[^src-probts] |

## 相关页面

- [[probts]] / [[source-probts]]
- [[ar-vs-nar-decoding]]
- [[generative-time-series-forecasting]]
- [[csdi]] / [[timegrad]]
- [[instance-normalization]]

[^src-probts]: [[source-probts]]
