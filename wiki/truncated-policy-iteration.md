---
title: "截断策略迭代"
type: technique
tags:
  - reinforcement-learning
  - policy-iteration
  - value-iteration
  - trade-off
created: 2026-04-27
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# 截断策略迭代

截断策略迭代（Truncated Policy Iteration）在每轮只做有限步策略评估，再执行策略改进，用于折中评估精度与计算成本。[^src-chapter-4-value-iteration-and-policy-iteration]

## 算法描述

与完整策略迭代的区别：策略评估不运行到完全收敛，而是仅运行 $k$ 步值更新：

$$v_0 \xrightarrow{\text{Policy Evaluation (k steps)}} v_k \xrightarrow{\text{Policy Improvement}} \pi'$$

当 $k=1$ 时退化为 [[value-iteration|值迭代]]；当 $k \to \infty$ 时逼近 [[policy-iteration|策略迭代]]。[^src-chapter-4-value-iteration-and-policy-iteration]

## 统一视角

该框架揭示了「评估精度—每轮成本—总迭代轮数」三者之间的工程化权衡：

| 评估步数 $k$ | 极端情况 | 每轮成本 | 收敛轮数 |
|-------------|---------|---------|---------|
| $k=1$ | [[value-iteration\|值迭代]] | 低 | 多 |
| $k$ 中等 | 截断策略迭代 | 适中 | 适中 |
| $k \to \infty$ | [[policy-iteration\|策略迭代]] | 高 | 少 |

值迭代和策略迭代是同一算法族在不同评估深度下的两个端点。[^src-chapter-4-value-iteration-and-policy-iteration]

## 收敛性

由于策略改进引理仍适用，策略在每次改进后性能单调不降。当每轮评估步数固定时，算法最终收敛到最优策略，但收敛速度介于值迭代和策略迭代之间。

## 适用场景

- **计算资源有限**：限制评估步数以控制成本
- **策略改进阶段更关键**：高频改进 + 低精度评估
- **在线设置**：希望策略持续更新而非等待完整评估

[^src-chapter-4-value-iteration-and-policy-iteration]: [[source-chapter-4-value-iteration-and-policy-iteration]]
