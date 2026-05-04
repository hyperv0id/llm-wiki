---
title: "Instance Normalization (RevIN)"
type: technique
tags:
  - time-series-forecasting
  - normalization
  - distribution-shift
created: 2026-04-28
last_updated: 2026-04-29
source_count: 2
confidence: medium
status: active
---

# Instance Normalization (RevIN)

实例归一化（Instance Normalization），也称为可逆归一化（RevIN），是时序预测中处理分布漂移的常用技术[^src-cyclenet]。

## 背景

时序数据的统计特性（如均值）往往随时间变化，这称为分布漂移（distributional shifts）。这会导致在历史训练集上训练的模型在应用于未来数据时性能下降[^src-cyclenet]。

## 方法

RevIN 通过在模型输入和输出步骤中移除和恢复统计特性来解决这个问题[^src-cyclenet]：

### 前向归一化

$$x_{t-L+1:t} = \frac{x_{t-L+1:t} - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

其中 $\mu$ 和 $\sigma$ 是输入窗口的均值和标准差，$\epsilon$ 是数值稳定的小常数[^src-cyclenet]。

### 后向反归一化

$$\bar{x}_{t+1:t+H} = \bar{x}_{t+1:t+H} \times \sqrt{\sigma^2 + \epsilon} + \mu$$

## CycleNet 中的使用

CycleNet 采用不含可学习仿射参数的 RevIN 版本[^src-cyclenet]：

```python
# 前向
xt = (xt - mean) / (std + eps)
# 后向
pred = pred * (std + eps) + mean
```

## 消融实验结果

| 数据集 | 有 RevIN | 无 RevIN | 影响 |
|--------|---------|----------|------|
| ETTh2 | 显著提升 | - | 分布漂移严重 |
| Weather | 显著提升 | - | 分布漂移严重 |
| Solar | 性能下降 | - | 夜间零值影响均值计算 |

在大多数情况下 RevIN 带来更好的性能，但在 Solar 数据集上由于夜间零值段的影响导致性能下降[^src-cyclenet]。

## 与其他模型的关系

RevIN 已被多种主流模型采用：iTransformer、PatchTST、SparseTSF 等[^src-cyclenet]。

## 相关页面

- [[unica|UniCA]] — 统一的协变量适应框架，被设计为RevIN的广义扩展，可处理异构协变量（分类/图像/文本）而不仅是数值序列的分布漂移[^src-unica]
- [[unified-covariate-adaptation]] — 统一协变量适应概念，RevIN 是其处理分布漂移的子技术

## 引用

[^src-unica]: [[source-unica]]
[^src-cyclenet]: [[source-cyclenet]]