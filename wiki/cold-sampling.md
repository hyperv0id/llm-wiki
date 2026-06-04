---
title: "Cold Sampling"
type: technique
tags:
  - diffusion-models
  - generalized-diffusion
  - sampling-algorithm
  - ode-solver
created: 2026-06-04
last_updated: 2026-06-04
source_count: 1
confidence: medium
status: active
---

# Cold Sampling

**Cold Sampling** 是 Cold Diffusion（Bansal et al., 2022）提出的针对"广义扩散模型"的采样算法，在 DYffusion（NeurIPS 2023）中被关键性地用于从动力学信息扩散模型中生成概率预测[^src-dyffusion]。

## 定义

Cold Sampling 是 DDIM 采样在广义扩散模型上的推广。给定退化算子 $\mathcal{D}$ 和恢复网络 $\mathcal{R}_\theta$，Cold Sampling 的迭代更新为：

1. 估计目标：$\hat{s}^{(0)} = \mathcal{R}_\theta(s^{(n)}, n)$
2. 向前一步然后回退：$s^{(n+1)} = s^{(n)} + \mathcal{D}(\hat{s}^{(0)}, n+1) - \mathcal{D}(\hat{s}^{(0)}, n)$

核心思想：$\mathcal{D}(\hat{s}^{(0)}, n+1) - \mathcal{D}(\hat{s}^{(0)}, n)$ 作为误差修正项，使迭代沿正确的轨迹前进[^src-dyffusion]。

## DYffusion 中的应用

在 DYffusion 中，冷采样具体表现为预测-插值的交替过程[^src-dyffusion]：

1. $x_{t+h}^{(n)} = F_\theta(\hat{x}_{t+i_n}, i_n)$ — 预测最终快照
2. $\hat{x}_{t+i_{n+1}} = \hat{x}_{t+i_n} + \mathcal{I}_\phi(x_t, x_{t+h}^{(n)}, i_{n+1}) - \mathcal{I}_\phi(x_t, x_{t+h}^{(n)}, i_n)$ — 插值修正

## 理论分析

DYffusion 证明了 Cold Sampling 可以从 ODE 视角理解：整个过程等价于用 Euler 方法求解一个描述动力系统演化的 ODE[^src-dyffusion]。单步离散误差 $\|e\|_2$ 被界为 $O(\Delta s)$，而对于 Naive Sampling（无误差修正项），离散误差不受一阶界约束，解释了 Cold Sampling 在实验中大幅优于 Naive Sampling 的原因（SST CRPS 0.181 vs 0.681）[^src-dyffusion]。

## 与 DDIM 的关系

Cold Sampling 是 DDIM 的推广：当退化算子 $\mathcal{D}$ 为高斯加噪时，Cold Sampling 退化为 DDIM[^src-dyffusion]。

[^src-dyffusion]: [[source-dyffusion]]
