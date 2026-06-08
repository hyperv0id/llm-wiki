---
title: "Gaussian Process Priors in Flow Matching"
type: concept
tags:
  - flow-matching
  - gaussian-process
  - prior-distribution
  - optimal-transport
  - time-series
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# Gaussian Process Priors in Flow Matching

**高斯过程先验流匹配**是在 [[flow-matching|条件流匹配]] 框架中使用非各向同性高斯过程作为先验分布的技术[^src-tsflow]。由 TSFlow 首次引入时间序列预测领域[^src-tsflow]。

## 动机

传统 CFM 和扩散模型使用各向同性高斯先验 $q_0 = \mathcal{N}(0, I)$。然而，这忽略了一个事实：时间序列数据具有时间相关性——相邻时间点的值通常相似[^src-tsflow]。通过使用 GP 先验 $q_0 = \mathcal{GP}(0, K)$，我们可以将这种时序结构编码到先验中，减少先验与数据分布之间的 Wasserstein 距离，从而简化流匹配的学习问题[^src-tsflow]。

## 三种核函数

TSFlow 探索了三种 GP 核函数，每种反映不同的数据特征[^src-tsflow]：

| 核函数 | 定义 | 特性 | 适用场景 |
|--------|------|------|----------|
| **SE** (平方指数) | $K(\tau, \tau') = \exp(-d^2/2\ell^2)$ | 无限平滑 | 高度平滑的序列 |
| **OU** (Ornstein-Uhlenbeck) | $K(\tau, \tau') = \exp(-|d|/\ell)$ | Brownian 运动相关 | 粗糙结构 |
| **PE** (周期) | $K(\tau, \tau') = \exp(-2\sin^2(d)/\ell^2)$ | 捕获周期模式 | 周期性数据 |

其中 $d = \tau - \tau'$，$\ell$ 为长度尺度参数。

## 与最优传输的协同

GP 先验与最优传输耦合协同作用[^src-tsflow]：

1. **GP 先验** 将先验分布拉近数据分布（降低 W₂ 距离）
2. **OT 耦合** 在训练时配对先验样本与数据样本，进一步拉直概率路径

图 6 显示，PE 核对周期数据的 W₂ 距离降低最为显著，且随序列长度增加的退化最小[^src-tsflow]。

## 条件化：GP 回归先验

在条件预测设置中，TSFlow 使用 GP 回归 (GPR) 分析计算条件先验[^src-tsflow]：

$$
q_0(x_0^f \mid y^p) = \mathcal{N}(\mu_{f|p}, \Sigma_{f|p})
$$

其中：
- $\mu_{f|p} = \Sigma_{fp} \Sigma_{pp}^{-1} y^p$
- $\Sigma_{f|p} = \Sigma_{ff} - \Sigma_{fp} \Sigma_{pp}^{-1} \Sigma_{pf}$

协方差矩阵在推理时仅需计算一次，额外开销仅为一个向量-矩阵乘法[^src-tsflow]。

## 实验结果

- 非各向同性 GP 先验在 4 NFE 下超越各向同性先验在 16 NFE 下的性能[^src-tsflow]
- PE 核在无条件 LPS 评估中 6/8 数据集最优[^src-tsflow]
- OU 核条件 GPR 在概率预测（CRPS）中 7/8 数据集超越扩散基线[^src-tsflow]
- GPR 训练时间与各向同性先验几乎无差异（仅多一次矩阵-向量乘法）[^src-tsflow]

## 相关页面

- [[tsflow]] — TSFlow 模型
- [[flow-matching]] — Flow Matching 理论基础
- [[gaussian-process-regression]] — 高斯过程回归
- [[optimal-transport]] — 最优传输理论
- [[conditional-prior-sampling]] — 条件先验采样技术
- [[guided-generation]] — 引导生成

[^src-tsflow]: [[source-tsflow]]
