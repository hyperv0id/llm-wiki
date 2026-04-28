---
title: "On-policy 与 Off-policy"
type: analysis
tags:
  - reinforcement-learning
  - on-policy
  - off-policy
created: 2026-04-27
last_updated: 2026-04-27
source_count: 1
confidence: high
status: active
---

# On-policy 与 Off-policy

On-policy 与 Off-policy 的区别在于：行为策略与目标策略是否相同。[^src-chapter-7-temporal-difference-methods]

[[sarsa-algorithm|Sarsa]] 属于 on-policy：采样与更新都围绕同一策略；[[q-learning-algorithm|Q-learning]] 属于 off-policy：可用探索行为采样，但按贪心目标更新。[^src-chapter-7-temporal-difference-methods]

off-policy 的优势是数据复用能力更强；on-policy 常在稳定性与解释性上更直接。[^src-chapter-7-temporal-difference-methods]

[^src-chapter-7-temporal-difference-methods]: [[source-chapter-7-temporal-difference-methods]]
