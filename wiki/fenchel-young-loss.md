---
title: "Fenchel-Young Loss"
type: technique
tags:
  - loss-function
  - convex-optimization
  - end-to-end-learning
  - bregman-divergence
created: 2025-07-25
last_updated: 2025-07-25
source_count: 1
confidence: medium
status: active
---

# Fenchel-Young Loss

Fenchel-Young（FY）损失是一种用于学习结构化预测的损失函数，由 Blondel et al. (2020) 系统提出[^src-wardropnet]，在 [[wardropnet]] 中被用于训练 COAML 管道。它基于 Fenchel-Young 不等式，在凸函数 Ω 与其 Fenchel 共轭 Ω* 之间建立联系，为通过组合层进行端到端学习提供了理论基础。

## 定义

给定正则化凸函数 Ω 及其 Fenchel 共轭 Ω*(θ) = max_{y∈Ȳ} θ^⊤ y − Ω(y)，Fenchel-Young 损失定义为：

L_Ω(θ, ȳ) = Ω*(θ) + Ω(ȳ) − θ^⊤ ȳ

该损失满足三个关键性质[^src-wardropnet]：
1. **非负性**：L_Ω(θ, ȳ) ≥ 0（由 Fenchel-Young 不等式保证）
2. **零损失条件**：L_Ω(θ, ȳ) = 0 ⟺ ŷ_Ω(θ) = ȳ（预测等于目标）
3. **凸性**：θ ↦ L_Ω(θ, ȳ) 是凸函数

## 梯度

若 Ω* 在 θ 处可微，由 Danskin 引理：

∇_θ L_Ω(θ, ȳ) = ∇Ω*(θ) − ȳ = ŷ_Ω(θ) − ȳ

这意味着梯度仅是预测均衡与目标均衡之差，极为简洁[^src-wardropnet]。

## 与 Bregman 散度的关系

若正则化函数 ψ 是 Legendre 型，则 FY 损失是 Bregman 散度 D_ψ(ȳ, ŷ) 的凸上界[^src-wardropnet]：

0 ≤ D_ψ(ȳ, ŷ) ≤ L_Ω(θ, ȳ)

其中 ŷ = ŷ_Ω(θ)。当损失最小时两者相等。这意味着最小化 FY 损失等价于在 Bregman 散度几何下最小化目标与预测的距离。

在扰动正则化的特殊情形下（Ω = F*），Bregman 散度与 FY 损失完全等价：L_Ω(θ, ȳ) = D_Ω(ȳ, ŷ)[^src-wardropnet]。

## 在 WardropNet 中的应用

[[wardropnet|WardropNet]] 用 FY 损失训练 [[combinatorial-optimization-augmented-machine-learning|COAML]] 管道。正则化 Ω 的选择决定了损失的具体形式和梯度计算方式[^src-wardropnet]：

| 正则化 | ∇_θ L 计算 | 特点 |
|--------|------------|------|
| 欧几里得 | ŷ = proj_{Ȳ}(θ)，梯度 = ŷ − ȳ | 需凸二次优化 |
| 扰动 | ŷ = E[arg max (θ+Z)^⊤ y]，MC 估计 | Bregman = FY loss |
| 分段常数 | 扩展网络上求解 MCFP | 贴近真实队列物理 |

[^src-wardropnet]: [[source-wardropnet]]
