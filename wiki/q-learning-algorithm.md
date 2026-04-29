---
title: "Q-learning 算法"
type: technique
tags:
  - reinforcement-learning
  - td-learning
  - off-policy
created: 2026-04-27
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# Q-learning 算法

Q-learning 是 off-policy 的 TD 控制算法，直接逼近最优动作值函数 $q_*$。由 Watkins 在 1989 年提出，是强化学习中最具影响力的算法之一。[^src-chapter-7-temporal-difference-methods]

## 核心公式

Q-learning 的更新规则为：

$$q(S_t, A_t) \leftarrow q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a'} q(S_{t+1}, a') - q(S_t, A_t) \right]$$

其中：
- $S_t, A_t$ 是当前状态和动作
- $R_{t+1}$ 是即时奖励
- $S_{t+1}$ 是下一状态
- $\alpha \in (0,1]$ 是学习率
- $\gamma \in [0,1)$ 是折扣因子

## Off-policy 特性

关键特征是**行为策略与目标策略可分离**：

- **行为策略**：控制智能体实际采取什么动作（例如 ε-greedy）
- **目标策略**：学习的贪婪策略，直接逼近 $\pi_*$

这使得 Q-learning 可以复用其他策略产生的历史数据，这一特性对经验回放（Experience Replay）和 DQN 等深度强化学习方法至关重要。[^src-chapter-7-temporal-difference-methods]

## 收敛条件

Q-learning 在下列条件下以概率 1 收敛到最优 $q_*$：
- 学习率序列 $\{\alpha_t\}$ 满足 $\sum_t \alpha_t = \infty$ 且 $\sum_t \alpha_t^2 < \infty$
- 所有状态-动作对被无限频繁访问（探索性条件）

## 与 Sarsa 的对比

| 特性 | Q-learning | [[sarsa-algorithm|Sarsa]] |
|------|------------|------|
| 更新方式 | Off-policy | On-policy |
| 更新目标 | $\max_{a'} q(s', a')$ | $q(s', a')$ |
| 算法类型 | 最优策略学习 | 当前策略学习 |
| 安全性 | 可能更大胆 | 更保守 |

Q-learning 直接学习最优动作值函数，而 Sarsa 学习当前行为策略的动作值函数。这意味着 Sarsa 在奖励中含有负面惊喜（如悬崖行走）时表现更安全。[^src-chapter-7-temporal-difference-methods]

## 变体与扩展

- **Double Q-learning**：使用两个 Q 函数减轻最大化偏差
- **Dyna-Q**：结合模型学习和规划
- **DQN**：使用深度神经网络逼近 Q 函数，配合经验回放和目标网络

## 算法伪代码

```
初始化 q(s,a) 为任意值
对每个回合：
    初始化状态 s
    使用 q 的策略在 s 选择动作 a（如 ε-greedy）
    循环：
        执行动作 a，观察 r, s'
        从 s' 使用 q 的策略选择动作 a'（如 ε-greedy）
        q(s,a) ← q(s,a) + α[r + γ·max q(s',a') - q(s,a)]
        s ← s'; a ← a'
    直到 s 为终止状态
```

[^src-chapter-7-temporal-difference-methods]: [[source-chapter-7-temporal-difference-methods]]
