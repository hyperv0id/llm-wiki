---
title: "截断策略迭代"
type: technique
tags:
  - reinforcement-learning
  - policy-iteration
  - value-iteration
  - trade-off
created: 2026-04-27
last_updated: 2026-04-27
source_count: 1
confidence: high
status: active
---

# 截断策略迭代

截断策略迭代在每轮只做有限步策略评估，再执行策略改进，用于折中评估精度与计算成本。[^src-chapter-4-value-iteration-and-policy-iteration]

当评估步数为 1 时退化为 [[value-iteration|值迭代]]；评估步数足够大时逼近 [[policy-iteration|策略迭代]]。[^src-chapter-4-value-iteration-and-policy-iteration]

它提供了统一视角：值迭代与策略迭代是同一算法族在不同评估深度下的两个端点。[^src-chapter-4-value-iteration-and-policy-iteration]

[^src-chapter-4-value-iteration-and-policy-iteration]: [[source-chapter-4-value-iteration-and-policy-iteration]]
