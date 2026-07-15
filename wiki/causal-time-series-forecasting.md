---
title: "Causal Time-Series Forecasting (Interventional & Counterfactual)"
type: concept
tags:
  - time-series
  - causal-inference
  - counterfactual
  - interventional
  - structural-causal-model
created: 2026-06-08
last_updated: 2026-07-14
source_count: 2
confidence: medium
status: active
---

# 因果时间序列预测（干预与反事实）

**因果时间序列预测**指超越纯观测（相关性外推）的预测范式，目标是回答关于多变量动力系统的因果"what-if"问题[^src-doflow]。大多数现代预测器是观测式的：它们从历史中学习相关性并外推，但无法回答因果查询[^src-doflow]。

## 三类查询

在定义于因果 DAG 上的结构因果模型 $X_{i,t} := f_i(X_{i,t^-}, X_{\text{pa}(i),t^-}, U_{i,t})$ 框架下，可区分三类预测查询[^src-doflow]：

| 查询 | 问题 | 分布 |
|------|------|------|
| 观测（observational） | 在生成历史的同一条件下未来会如何？ | $p(X_{\tau+1:T}\mid x_{1:\tau})$ |
| 干预（interventional） | 在计划性地修改某些变量 $\text{do}(X_I:=\gamma_I)$ 后预测如何变化？ | $p(X_{\tau+1:T}\mid x_{1:\tau}, \text{do}(X_I:=\gamma_I))$ |
| 反事实（counterfactual） | 若当初对这条已观测轨迹施加不同干预，它会变成什么样？ | $p(X^{CF}_{\tau+1:T}\mid x_{1:\tau}, x^F_{\tau+1:T}, \text{do}(X_I:=\gamma_I))$ |

当干预调度 $I=\emptyset$ 时，干预预测退化为标准观测预测[^src-doflow]。反事实预测遵循 Pearl 的 abduction–action–prediction 三步法，并需要条件于已观测的事实轨迹以推断未观测的外生因素[^src-doflow]。

## 与相邻问题的区分

实践者常将此范式与两条互补线结合[^src-doflow]：
- **因果效应估计（causal effect estimation）**：量化外部行动如何改变短期期望结果（如治疗效应 $\tau_t = E[Y_t\mid A_{t-1}=j] - E[Y_t\mid A_{t-1}=k]$），通常聚焦离散、固定时刻的行动[^src-doflow]。
- **因果发现（causal discovery）**：从观测时间序列中恢复因果 DAG[^src-doflow]。

截至 2026 年，已有方法主要关注离散固定时刻行动并估计短期期望结果差异；对完整**反事实轨迹**的生成式建模仍是新兴方向[^src-doflow]。

## 假设

干预与反事实分布的良定义通常依赖标准因果假设：无混杂（causal sufficiency）、一致性（consistency）、正性（positivity）、无干涉（no interference）[^src-doflow]。在单调 SCM 下，可建立反事实的逐点恢复保证（见 [[causal-counterfactual-recovery]]）[^src-doflow]。

## 方法

- [[doflow|DoFlow]]（ICLR 2026）：用连续归一化流在已知因果 DAG 上统一三类查询，是该范式下少数支持反事实轨迹生成的框架之一[^src-doflow]。
- [[causalx|CausalX]]（ICML 2026）：另一条因果路线——不回答干预/反事实查询，而是用多源因果约束（Granger、do-calculus、TDMI、VAE）+ 扩散精炼学习因果启发的动态图，以提升预测精度和可解释性[^src-causalx]。

## 链接

- [[doflow]] — DoFlow，因果 DAG 上的流式生成预测
- [[causal-counterfactual-recovery]] — abduction–action–prediction 与反事实恢复理论
- [[e2-cstp]] — 因果时空预测（混杂消除路线）
- [[causalx]] — 因果图学习路线（多源约束 + 扩散精炼）

[^src-doflow]: [[source-doflow]]
[^src-causalx]: [[source-causalx]]

