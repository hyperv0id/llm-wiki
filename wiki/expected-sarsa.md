---
title: "Expected Sarsa"
type: technique
tags:
  - reinforcement-learning
  - td-learning
  - variance-reduction
created: 2026-04-27
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# Expected Sarsa

Expected Sarsa 是 Sarsa 的变体，用「下一状态动作值的期望」替代 Sarsa 中的单次采样值，从而降低更新方差。[^src-chapter-7-temporal-difference-methods]

## 核心公式

Expected Sarsa 的更新规则为：

$$q(S_t, A_t) \leftarrow q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \sum_{a'} \pi(a'|S_{t+1}) q(S_{t+1}, a') - q(S_t, A_t) \right]$$

其中 $\sum_{a'} \pi(a'|S_{t+1}) q(S_{t+1}, a')$ 是对所有可能动作 $a'$ 的期望值，而非对单次采样 $A_{t+1}$ 的依赖。

## 方差降低原理

标准 Sarsa 中 $q(S_{t+1}, A_{t+1})$ 包含两个随机源：$S_{t+1}$ 的环境随机性，和 $A_{t+1}$ 的策略随机性。Expected Sarsa 消除了策略随机性带来的方差，因此更新更平滑、学习曲线更稳定。在策略随机性较高时（如 $\varepsilon$ 较大），优势尤其明显。[^src-chapter-7-temporal-difference-methods]

## 与 Q-learning 的关系

有意思的是，Expected Sarsa 在行为策略为贪心策略时退化为 Q-learning：

$$\sum_{a'} \pi_{\text{greedy}}(a'|s') q(s', a') = q(s', \arg\max_a q(s', a)) = \max_{a'} q(s', a')$$

与 [[q-learning-algorithm|Q-learning]] 的区别：Expected Sarsa 使用策略期望，不受最大化偏差影响；Q-learning 使用最大化算子，可能产生过高估计。[^src-chapter-7-temporal-difference-methods]

## 优势

1. **低方差**：消除策略随机性的影响
2. **灵活的目标策略**：目标策略可以不等于行为策略
3. **无最大化偏差**：不需要 max 操作
4. **稳健性**：对学习率选择不敏感

[^src-chapter-7-temporal-difference-methods]: [[source-chapter-7-temporal-difference-methods]]
