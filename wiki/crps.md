---
title: "Continuous Ranked Probability Score (CRPS)"
type: concept
tags:
  - evaluation-metric
  - probabilistic-forecasting
  - scoring-rule
created: 2026-07-25
last_updated: 2026-07-25
source_count: 1
confidence: low
status: active
---

# Continuous Ranked Probability Score (CRPS)

**CRPS**（连续分级概率评分）是一种**严格合适评分规则**（proper scoring rule），用于评估概率预测的校准度和锐度。其核心性质：CRPS 在预测分布等于真实数据分布时达到最小值，因此能同时惩罚过度自信（分布过窄但偏离观测）和过度保守（分布过宽）[^src-timegrad]。

## 定义

对累积分布函数 $F$ 和观测值 $x$：

$$\text{CRPS}(F, x) = \int_{\mathbb{R}} (F(z) - \mathbb{I}\{x \leq z\})^2 dz$$

其中 $\mathbb{I}\{x \leq z\}$ 是指示函数（$x \leq z$ 时为 1，反之为 0）[^src-timegrad]。CRPS 将 CDF 和观测分布在所有阈值 $z$ 上的累积差距平方后积分——本质上是预测 CDF 与理想阶跃 CDF 之间的 $L^2$ 距离。

## 经验 CRPS

当预测分布只能通过 $S$ 个样本 $X_s \sim F$ 获得时，使用经验 CDF $\hat{F}(z) = \frac{1}{S}\sum_{s=1}^{S} \mathbb{I}\{X_s \leq z\}$ 直接计算[^src-timegrad]。[[timegrad|TimeGrad]] 使用 $S=100$ 条独立轨迹的经验 CDF 进行 CRPS 评估。

## CRPS_sum 变体

在多变量时间序列中，**CRPS_sum** 先将所有 $D$ 个维度求和得到总量标量（对真实数据和每个采样分别求和），再对该总量分布的 $\hat{F}_{\text{sum}}$ 计算 CRPS[^src-timegrad]。这确保 CRPS_sum 仍是严格合适评分规则。CRPS_sum 被 [[timegrad|TimeGrad]] 及后续大量概率预测方法（[[diffstg|DiffSTG]]、[[csdi|CSDI]] 等）作为主要评估指标。

## 与其他指标对比

与基于似然的指标（NLL）不同，CRPS 不需要模型输出解析分布形式——只要能从分布中采样即可计算。这使其特别适合以采样为核心推理机制的扩散模型和基于模拟的方法[^src-timegrad]。

## 关联页面
- [[crps-autoregressive-finetuning]] — CRPS 作为扩散生成式模型的微调目标 (Swift, 2025)

[^src-timegrad]: [[source-timegrad]]
