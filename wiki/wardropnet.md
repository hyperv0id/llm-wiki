---
title: "WardropNet"
type: entity
tags:
  - traffic-prediction
  - combinatorial-optimization
  - equilibrium-learning
  - deep-learning
created: 2025-07-25
last_updated: 2025-07-25
source_count: 1
confidence: medium
status: active
---

# WardropNet

WardropNet 是一种 [[combinatorial-optimization-augmented-machine-learning|COAML]]（Combinatorial Optimization-Augmented Machine Learning）管道，将神经网络统计模型与 [[wardrop-equilibrium|Wardrop 均衡（WE）]] 组合层结合，用于交通流预测[^src-wardropnet]。发表于 ICLR 2025。

## 架构

管道由两部分组成：

1. **统计模型 φ_w**：以上下文信息 x（弧属性、节点属性）为输入，预测延迟函数的参数化 θ = φ_w(x)。支持 FNN 和 GNN 两种实现[^src-wardropnet]。

2. **均衡层 ŷ_Ω(θ)**：接收延迟参数 θ、交通网络拓扑和 OD 对，求解 Wardrop 均衡得到预测交通流。均衡问题形式化为 `arg max_y θ^⊤ y − Ω(y)`，其中 Ω 为正则化项[^src-wardropnet]。

端到端训练使用 [[fenchel-young-loss|Fenchel-Young 损失]]：`L_Ω(θ, ȳ) = Ω*(θ) + Ω(ȳ) − θ^⊤ ȳ`，该损失是 Bregman 散度的凸上界[^src-wardropnet]。

## 三种变体

| 变体 | 正则化 | 延迟函数形式 | 均衡求解 |
|------|--------|-------------|---------|
| ER | 欧几里得 | y-intercept only: ℓ_a = −θ_a + ȳ_a | 凸二次优化 |
| CL | 扰动正则化 | 分段常数（Vickrey 队列模型） | 扩展网络 MCFP |
| PL | 扰动正则化 | 多项式：ℓ_a = θ_{0,a} + θ_{1,a}ȳ_a | WE + MC 梯度估计 |

CL 在所有场景中表现最优，可能因为分段常数延迟更贴近真实交通的队列物理特性[^src-wardropnet]。

## 关键结果

- 在柏林真实场景上，CL 管道较 FNN 基线 MAE 降低 72%[^src-wardropnet]
- 即使在高熵场景（上下文-流量高相关，纯 ML 有利），COAML 管道仍优于纯 ML
- 结构分析显示 COAML 能捕捉主干道高流量 + 支路低流量的组合模式，而纯 ML 仅预测均值

## 代码

开源：<https://github.com/tumBAIS/ML-CO-pipeline-TrafficPrediction>

[^src-wardropnet]: [[source-wardropnet]]
