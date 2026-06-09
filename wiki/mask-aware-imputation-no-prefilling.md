---
title: "Mask-Aware Imputation (No Pre-filling)"
type: concept
tags:
  - data-imputation
  - spatio-temporal
  - mask-aware
  - inductive-bias
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Mask-Aware Imputation (No Pre-filling)

**无预填充的掩码感知填补**指一类时间序列/时空填补方法，它们不把缺失位置初始化为占位值（零、均值等）后再当作完整数据建模，而是用可学习表示直接表征缺失值，并用缺失掩码主动调节信息聚合[^src-maginet]。

## 动机：预填充的危害

多数深度填补方法（[[grin|GRIN]]、BRITS、GAIN、[[csdi|CSDI]] 等）采用**零预填充**初始化缺失值并用掩码矩阵记录其位置，再把'补完'的数据视作完整数据做特征学习[^src-maginet]。但用占位值（NaN→0）初始化会向特征学习注入不可控噪声并误导模型；[[maginet|MagiNet]] 在 Seattle 数据集上实证：带预填充的填补性能显著劣于不带预填充[^src-maginet]。此外，在预填充数据上捕获时空相关性会忽略内在动态变化，导致动态/连续缺失位置出现**过平滑插值**（参见 [[over-smoothing-in-gnns]]）[^src-maginet]。

## 核心机制

[[maginet|MagiNet]] 给出的无预填充实现包含两个要素[^src-maginet]：

1. **可学习缺失嵌入**：缺失矩阵经可学习嵌入层映射为 Z_u，按掩码与观测嵌入组合 X_p = X_o ⊙ M + Z_u ⊙ (1−M)，缺失位置由可学习表示占位而非零。
2. **掩码乘入注意力**：把掩码 M 乘入注意力分数（C = Softmax(M ⊙ A)V），屏蔽缺失值对观测的污染，并把注意力作为权重注入图卷积核以动态调整聚合系数。

## 与'缺失感知输入'的区别

[[message-passing-imputation|消息传递填补]] 中的 [[grin|GRIN]] 式做法是把缺失掩码**拼接**到（已预填充的）输入特征中，让网络区分观测值与填补值——掩码是辅助特征，但预填充仍然发生。无预填充范式更激进：缺失位置自始至终不被赋予占位数值，掩码进一步用于阻断而非仅标记[^src-maginet]。

## 相关方法

- [[maginet]] — 首个明确论证'预填充有害'并用可学习缺失嵌入移除预填充的时空填补方法（arXiv 2024）
- [[grin]] / [[pristi]] / [[csdi]] — 采用预填充的对照方法

[^src-maginet]: [[source-maginet]]

