---
title: "Linear Predictive Score (LPS)"
type: technique
tags:
  - evaluation
  - synthetic-data
  - time-series
  - probabilistic-forecasting
  - generative-model
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: medium
status: active
---

# Linear Predictive Score (LPS)

**Linear Predictive Score (LPS)** 是 [[tsdiff|TSDiff]] 论文提出的合成时间序列质量指标：在生成模型的合成样本上训练**线性 ridge 回归预测器**，再在真实测试集上计算 **CRPS**（对点预测器等价于 0.5-分位损失 / Normalized Deviation）[^src-prs]。设定为经典的 **train-on-synthetic, test-on-real**[^src-prs]。

## 动机

图像域可用 Inception 类固定下游网络做 FID/IS 类指标；时序生成文献常自选下游网络架构，结果对架构、初始化乃至框架敏感，且每次度量要完整训练深度模型[^src-prs]。LPS 用闭式可解的 ridge 回归降低方差与成本，并直接度量**预测有用性**而非仅分布距离[^src-prs]。

## 定义与实现

1. 用生成模型采样 $N$ 条合成序列（TSDiff 实验用 10,000 条）[^src-prs]。
2. 以与主实验一致的 context / prediction 长度，在合成数据上拟合 ridge（如 scikit-learn 默认正则）[^src-prs]。
3. 在对应真实测试划分上计算 CRPS，该值即为 LPS（**越低越好**）[^src-prs]。

## 实证角色

在 Solar / Electricity / Traffic 等 8 个基准上，TSDiff 的 LPS 显著优于 TimeVAE 与 TimeGAN；辅以 DeepAR、Transformer 的 train-on-synthetic 结果，进一步支持“样本保留关键预测结构”的结论[^src-prs]。后续 [[tsflow|TSFlow]] 继续采用 LPS 比较不同 GP 先验核下的无条件生成质量[^src-prs]。

## 局限

- 只探测**线性可利用**的预测信号，可能低估仅对深度模型有用的高阶结构[^src-prs]。
- 依赖 context/horizon 与归一化协议，跨论文比较需对齐协议[^src-prs]。
- 不等价于似然或样本多样性的完整刻画，宜与强下游预测器及定性样本对照[^src-prs]。

## 相关页面

- [[tsdiff]] — 提出并主用 LPS 的模型
- [[source-prs]] — 原始定义与表格
- [[tsflow]] / [[source-tsflow]] — 后续沿用 LPS 的流匹配工作
- [[generative-time-series-forecasting]] — 生成式预测语境

[^src-prs]: [[source-prs]]
