---
title: "ε-greedy 策略"
type: technique
tags:
  - reinforcement-learning
  - exploration
  - exploitation
  - epsilon-greedy
created: 2026-04-27
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# ε-greedy 策略

ε-greedy 是平衡探索与利用的基本策略，以概率 $1-\varepsilon$ 选择当前贪心动作，以概率 $\varepsilon$ 均匀随机地探索所有动作。[^src-chapter-5-monte-carlo-methods]

## 动作选择规则

$$A_t = \begin{cases}
\arg\max_a q(S_t, a) & \text{以概率 } 1-\varepsilon \quad (\text{利用}) \\
\text{均匀随机动作} & \text{以概率 } \varepsilon \quad (\text{探索})
\end{cases}$$

## 参数选择

$\varepsilon$ 值决定探索程度：
- $\varepsilon = 0$：纯利用，可能陷入局部最优
- $\varepsilon = 1$：纯探索，无法积累最优知识
- 常用值：$\varepsilon = 0.1$ 或 $\varepsilon = 0.01$

该机制使无模型算法在不依赖 Exploring Starts 的情况下仍能覆盖动作空间，是 [[monte-carlo-methods-rl|MC 方法]]、[[sarsa-algorithm|Sarsa]] 与 [[q-learning-algorithm|Q-learning]] 的常见行为策略。[^src-chapter-5-monte-carlo-methods]

## 衰减策略

为平衡前期探索与后期收敛，常采用 $\varepsilon$ 衰减：
- **线性衰减**：$\varepsilon_t = \max(\varepsilon_{\min}, \varepsilon_{\text{start}} - \frac{t}{\tau} \cdot (\varepsilon_{\text{start}} - \varepsilon_{\min}))$
- **指数衰减**：$\varepsilon_t = \varepsilon_{\min} + (\varepsilon_{\text{start}} - \varepsilon_{\min}) \cdot e^{-t/\tau}$

衰减需确保 $\varepsilon$ 不衰减过快，以维持足够的探索性，保证收敛。

## 与 Softmax 的对比

| 特性 | ε-greedy | Softmax |
|------|----------|---------|
| 动作选择 | 除最优外等概率 | 按指数权重比例 |
| 探索粒度 | 粗糙 | 精细 |
| 计算开销 | 低 | 较高（需指数运算） |

Softmax（玻尔兹曼探索）可更精细地区分「稍差」和「很差」的动作，但计算开销更大。

## 局限性

ε-greedy 在探索时将最优动作和极差动作等概率对待，在大动作空间中探索效率低下。更先进的探索策略（如 UCB、Thompson 采样等）在复杂任务中通常表现更好。

[^src-chapter-5-monte-carlo-methods]: [[source-chapter-5-monte-carlo-methods]]
