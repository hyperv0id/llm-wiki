---
title: "Sarsa 算法"
type: technique
tags:
  - reinforcement-learning
  - td-learning
  - on-policy
created: 2026-04-27
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# Sarsa 算法

Sarsa 是 on-policy 的 TD 控制算法，算法名称来源于更新所依赖的五元组：$(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$。[^src-chapter-7-temporal-difference-methods]

## 核心公式

Sarsa 的标准更新规则为：

$$q(S_t, A_t) \leftarrow q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma q(S_{t+1}, A_{t+1}) - q(S_t, A_t) \right]$$

其中 $A_{t+1}$ 是智能体在下一状态 $S_{t+1}$ **实际采取的**动作，由当前行为策略（如 ε-greedy）产生。

## On-policy 特性

与 [[q-learning-algorithm|Q-learning]] 的关键区别：Sarsa 评估的是当前行为策略的动作值函数，而非直接逼近最优贪心目标。这意味着 Sarsa 学习的是**智能体实际执行的**策略，而非独立的最优策略。[^src-chapter-7-temporal-difference-methods]

## 算法伪代码

```
初始化 q(s,a) 为任意值
对每个回合：
    初始化状态 s
    使用 q 的策略选择动作 a（如 ε-greedy）
    循环：
        执行动作 a，观察 r, s'
        使用 q 的策略从 s' 选择动作 a'（如 ε-greedy）
        q(s,a) ← q(s,a) + α[r + γ·q(s',a') - q(s,a)]
        s ← s'; a ← a'
    直到 s 为终止状态
```

## Sarsa vs Q-learning 对比

| 特性 | Sarsa | [[q-learning-algorithm\|Q-learning]] |
|------|-------|-------------|
| 更新方式 | On-policy | Off-policy |
| 更新目标 | $R + \gamma q(s', a')$ | $R + \gamma \max_{a'} q(s', a')$ |
| 策略关系 | 评估当前行为策略 | 学习独立的最优策略 |
| 安全性 | **更保守**，考虑探索影响 | 更乐观 |
| 适用场景 | 在线交互优先 | 离线数据利用优先 |

## 安全性分析

Sarsa 在奖励中含有负面惊喜时（悬崖行走问题）表现更安全：Sarsa 在更新时考虑了探索可能带来的负面后果，因此会避开悬崖附近的「软」路径；Q-learning 只优化贪婪路径，在探索阶段可能走入危险区域。[^src-chapter-7-temporal-difference-methods]

## 变体

- **[[expected-sarsa|Expected Sarsa]]**：用期望替代采样，降低方差
- **n-step Sarsa**：使用 n 步回报，在 MC 和 TD 间折中
- **Sarsa(λ)**：使用资格迹（eligibility trace）整合多步信息

[^src-chapter-7-temporal-difference-methods]: [[source-chapter-7-temporal-difference-methods]]
