---
title: "Chapter 4: Value Iteration and Policy Iteration"
type: source-summary
tags:
  - reinforcement-learning
  - value-iteration
  - policy-iteration
  - dynamic-programming
created: 2026-04-27
last_updated: 2026-04-27
source_count: 1
confidence: high
status: active
---

# Chapter 4: Value Iteration and Policy Iteration

本章把第 3 章的最优性理论转化为可执行算法，系统比较 [[value-iteration|值迭代]]、[[policy-iteration|策略迭代]] 与 [[truncated-policy-iteration|截断策略迭代]]。[^src-chapter-4-value-iteration-and-policy-iteration]

值迭代直接对最优性算子做不动点迭代，每轮执行“贪心改进 + 值更新”；策略迭代则显式交替执行“策略评估 + 策略改进”，并利用策略改进引理保证性能单调不降。两者都收敛到最优值函数与最优策略，但计算开销分布不同：值迭代单轮更轻，策略迭代轮数通常更少。[^src-chapter-4-value-iteration-and-policy-iteration]

截断策略迭代在两者间连续插值：当每轮评估步数取 1 时退化为值迭代，取充分大时逼近策略迭代。该框架说明“评估精度—每轮成本—总迭代轮数”之间存在工程化权衡。[^src-chapter-4-value-iteration-and-policy-iteration]

本章的价值在于给出从理论到算法的第一条完整路径，为后续无模型方法（MC、TD）提供参照基线。[^src-chapter-4-value-iteration-and-policy-iteration]

[^src-chapter-4-value-iteration-and-policy-iteration]: [[source-chapter-4-value-iteration-and-policy-iteration]]
