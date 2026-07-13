---
title: "Instance Normalization (RevIN)"
type: technique
tags:
  - time-series-forecasting
  - normalization
  - distribution-shift
created: 2026-04-28
last_updated: 2026-07-13
source_count: 6
confidence: high
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

RevIN 已被多种主流模型采用：iTransformer、PatchTST、SparseTSF 等[^src-cyclenet]。**PatchTST** 在 patching 前对每个单变量序列做零均值单位方差归一化，预测后将均值和标准差加回，与 [[channel-independence|Channel Independence]] 配合使用 [^src-patchtst]。**[[tide|TiDE]]** 将 RevIN 作为可调超参（on/off），与 residual MLP 骨干和协变量路径一起在验证集上选择[^src-tide]。**[[nuwats|NuwaTS]]** 在插补场景中对每个变量先做 RevIN（**缺失值置零**后再归一化）以消除跨域幅度/分布差异，存储原始均值方差供反归一化[^src-nuwats]。

### ProbTS 对 RevIN vs 均值缩放的跨场景结论

[[probts|ProbTS]] 将归一化作为与 [[ar-vs-nar-decoding|AR/NAR 解码]]、分布估计并列的第三方法轴，并系统对比了长程点预测线常用的 RevIN 与短程概率线常用的**均值缩放**（mean scaling）[^src-probts]：

- **长程**：RevIN 显著改善多数模型（尤其 AR 概率模型），掩盖趋势引起的分布漂移与误差累积；ETTh1 上 GRU-NVP+RevIN 甚至可超过 PatchTST+RevIN。但在强季节、弱趋势的 Traffic 上 RevIN 有负作用，暗示其主要对冲的是趋势效应[^src-probts]。
- **短程概率**：RevIN 对 CSDI / TimeGrad / GRU-NVP 等不稳定占优；**均值缩放更可靠**。完全不做实例级归一化有时可行，但在 Wikipedia/Solar （TimeGrad）、Electricity（GRU-NVP）上会导致严重失败[^src-probts]。
- **NAR 概率**：RevIN 与 CSDI 类 NAR 不总是好搭档（如 Weather 上 CSDI+RevIN 差于 CSDI+Scaling），NAR 概率模型的有效归一化仍是开放问题[^src-probts]。

## 相关页面

- [[unica|UniCA]] — 统一的协变量适应框架，被设计为RevIN的广义扩展，可处理异构协变量（分类/图像/文本）而不仅是数值序列的分布漂移[^src-unica]
- [[unified-covariate-adaptation]] — 统一协变量适应概念，RevIN 是其处理分布漂移的子技术
- [[patchtst|PatchTST]] — 在 patching 前使用 RevIN 与 CI 配合[^src-patchtst]
- [[tide|TiDE]] — 将 RevIN 作为验证集可调超参[^src-tide]
- [[nuwats|NuwaTS]] — 插补基础模型，对缺失值置零后做 RevIN 消除跨域分布差异[^src-nuwats]
- [[probts|ProbTS]] — 统一基准下 RevIN vs 均值缩放的系统对比[^src-probts]
- [[ar-vs-nar-decoding]] — 与归一化交互的解码方案轴

## 引用

[^src-unica]: [[source-unica]]
[^src-cyclenet]: [[source-cyclenet]]
[^src-patchtst]: [[source-patchtst]]
[^src-nuwats]: [[source-nuwats]]
[^src-probts]: [[source-probts]]
[^src-tide]: [[source-tide]]