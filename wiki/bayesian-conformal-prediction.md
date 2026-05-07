---
title: "Bayesian Conformal Prediction"
type: concept
tags:
  - conformal-prediction
  - bayesian
  - uncertainty-quantification
created: 2026-05-07
last_updated: 2026-05-07
source_count: 1
confidence: medium
status: active
---

# Bayesian Conformal Prediction

Bayesian CP 是一类将贝叶斯原理与共形预测相结合的在线自适应方法。核心思想是通过对历史非一致性评分施加指数时间折扣来适应分布漂移：$D_t^T(i) = \beta^{t-1-i}$。这可以看作 Follow-The-Regularized-Leader (FTRL) 算法的一个实例。

## 优势
- 提供最坏情况下的动态遗憾界保证
- 在分布漂移下维持覆盖率

## 局限性
- 存在结构滞后——极端残差会导致冲击结束后区间迟迟不回缩（未校准的区间膨胀）
- 对历史模式的再利用效率低

## 关联方法
- [[sa-bcp]] — 通过时空解耦解决 Bayesian CP 的结构滞后问题

[^src-sa-bcp]: [[source-sa-bcp]]
