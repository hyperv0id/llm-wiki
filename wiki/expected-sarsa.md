---
title: "Expected Sarsa"
type: technique
tags:
  - reinforcement-learning
  - td-learning
  - variance-reduction
created: 2026-04-27
last_updated: 2026-04-27
source_count: 1
confidence: high
status: active
---

# Expected Sarsa

Expected Sarsa 用“下一状态动作值的期望”替代 Sarsa 中的单次采样值，从而降低更新方差。[^src-chapter-7-temporal-difference-methods]

更新目标写作 $r+\gamma\sum_{a'}\pi(a'|s')q(s',a')$，在策略随机性较高时通常比 Sarsa 更平滑。[^src-chapter-7-temporal-difference-methods]

它与 [[q-learning-algorithm|Q-learning]] 的区别在于：前者使用策略期望，后者使用最大化算子。[^src-chapter-7-temporal-difference-methods]

[^src-chapter-7-temporal-difference-methods]: [[source-chapter-7-temporal-difference-methods]]
