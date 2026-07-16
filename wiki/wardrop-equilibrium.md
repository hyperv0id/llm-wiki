---
title: "Wardrop Equilibrium"
type: concept
tags:
  - traffic-modeling
  - game-theory
  - equilibrium
  - network-flow
created: 2025-07-25
last_updated: 2025-07-25
source_count: 1
confidence: medium
status: active
---

# Wardrop Equilibrium

Wardrop 均衡（WE）是交通流建模中的核心概念，描述了一种状态：每个出行者选择的路径都是其个人旅行时间最短的路径，且没有任何出行者可以通过单方面改变路径来减少自己的旅行时间[^src-wardropnet]。这是交通分配中"用户均衡"（User Equilibrium）的形式化定义，由 Wardrop (1952) 提出，在 [[wardropnet]] 中被推广到非可分解延迟函数的情形。

## 形式化定义

给定有向图 D=(V,A)、OD 对 (o_j, d_j)_{j∈J} 和延迟函数向量 ℓ={ℓ_a}_{a∈A}，一个多商品流 y=(y^{(j)})_{j∈J} ∈ Y 是 WE，当且仅当：

对于所有 j∈J 和 y'^{(j)}∈Y^{(j)}，ℓ(ȳ)^⊤ y^{(j)} ≤ ℓ(ȳ)^⊤ y'^{(j)}[^src-wardropnet]

其中 ȳ = Σ_j y^{(j)} 为聚合流，延迟函数 ℓ_a: R_+ → R_+ 将弧 a 上的聚合流映射为旅行时间。

## 广义 Wardrop 均衡

[[wardropnet|WardropNet]] 将 WE 推广到非可分解延迟函数的情形，允许弧上延迟依赖全网流量（而非仅该弧流量），以捕捉交通流溢流效应[^src-wardropnet]。关键定理：

**定理**：当延迟函数 ℓ 从势函数 Φ 导出（ℓ_a = ∂Φ/∂y_a），WE 等价于凸优化问题 min_{y∈Y} Φ(ȳ)[^src-wardropnet]。若 Φ 严格凸，则 WE 存在且唯一。

## 计算 WE

求解 WE 的方法分为两类[^src-wardropnet]：

**解析方法**：
- Frank-Wolfe 算法：迭代求解 Beckmann 公式化的凸优化问题
- 逐次平均法（Successive Averages）：Frank-Wolfe 的启发式变体
- Bar-Gera 算法：按起点分解原问题
- 投影方法：基于变分不等式，投影到可行流集

**仿真方法**：
- MATSim：基于代理的协同进化交通仿真，通过每个代理独立优化其出行计划收敛到随机用户均衡

## 与 COAML 的关系

在 [[combinatorial-optimization-augmented-machine-learning|COAML]] 管道中，WE 的 Beckmann 公式化被改写为带正则化的形式：max_{y∈Ȳ} θ^⊤ y − Ω(y)，其中 θ=φ_w(x) 由神经网络预测，Ω 为正则化项。这使得均衡计算成为可微的神经网络层[^src-wardropnet]。

[^src-wardropnet]: [[source-wardropnet]]
