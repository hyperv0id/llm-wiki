---
title: "MDP 形式化定义"
type: concept
tags:
  - reinforcement-learning
  - mdp
  - markov-property
created: 2026-04-27
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# MDP 形式化定义

马尔可夫决策过程（Markov Decision Process，MDP）是强化学习的标准问题模型，用于描述「智能体—环境」交互的序贯决策问题。[^src-chapter-1-basic-concepts]

## 形式化定义

有限 MDP 可由五元组 $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$ 定义：

- **状态空间** $\mathcal{S}$：智能体可能处于的所有状态集合
- **动作空间** $\mathcal{A}$：智能体可以采取的所有动作集合
- **转移概率** $P(s'|s,a)$：在状态 $s$ 采取动作 $a$ 后转移到状态 $s'$ 的概率
- **奖励函数** $R(s,a,s')$ 或 $R(s,a)$：状态转移获得的奖励
- **折扣因子** $\gamma \in [0,1)$：衡量未来奖励的重要性

## 马尔可夫性质

MDP 的核心假设是**马尔可夫性质**：未来只依赖于当前状态和动作，与历史无关。形式化表示为：

$$p(s_{t+1}, r_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = p(s_{t+1}, r_{t+1} | s_t, a_t)$$

这意味着环境模型只需知道当前状态和动作即可预测下一状态和奖励，无需记录完整的历史轨迹。[^src-chapter-1-basic-concepts]

## 策略

策略 $\pi(a|s)$ 是从状态到动作概率分布的映射，定义了智能体的行为规则。固定策略后，MDP 退化为马尔可夫过程（MP），此时「控制」问题可分解为两步：

1. **[[policy-evaluation|策略评估]]**：计算给定策略下的状态值函数 $v_\pi(s)$
2. **策略改进**：根据值函数调整策略，逐步逼近最优策略

这种「评估 + 改进」的框架是 [[policy-iteration|策略迭代]] 等动态规划算法的理论基础。[^src-chapter-1-basic-concepts]

## 回报与值函数

智能体的目标是最大化期望累积折扣回报：

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

状态值函数 $v_\pi(s) = \mathbb{E}[G_t | S_t = s]$ 表示从状态 $s$ 出发遵循策略 $\pi$ 的期望回报；动作值函数 $q_\pi(s,a)$ 则考虑首步动作 $a$。这两个函数通过 [[bellman-equation|贝尔曼方程]] 相互关联。[^src-chapter-1-basic-concepts]

## 任务类型

- **回合制任务（Episodic）**：智能体与环境交互直到达到终止状态
- **持续任务（Continuing）**：智能体无限期地与环境交互

吸收态（absorbing state）技术可将回合制任务转换为统一的马尔可夫框架，便于后续算法推导。[^src-chapter-1-basic-concepts]

[^src-chapter-1-basic-concepts]: [[source-chapter-1-basic-concepts]]
