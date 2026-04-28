---
title: "Chapter 7: Temporal-Difference Methods"
type: source-summary
tags:
  - reinforcement-learning
  - td-learning
  - sarsa
  - q-learning
created: 2026-04-27
last_updated: 2026-04-27
source_count: 1
confidence: high
status: active
---

# Chapter 7: Temporal-Difference Methods

本章聚焦时序差分（TD）方法：在未知环境模型下，用采样轨迹增量逼近贝尔曼方程（或贝尔曼最优方程）的解。它连接了动态规划的递归结构与蒙特卡洛的采样思想。[^src-chapter-7-temporal-difference-methods]

章节先给出单步 TD 的核心对象：TD 目标 $r_{t+1}+\gamma V(S_{t+1})$ 与 TD 误差。更新仅发生在被访问状态（或状态-动作对）上，因此天然适配在线学习与持续任务。[^src-chapter-7-temporal-difference-methods]

随后展开控制算法族：Sarsa（on-policy）、Expected Sarsa（用期望替换采样降低方差）、n-step Sarsa（多步回报折中偏差/方差）和 Q-learning（off-policy，使用 $\max$ 目标逼近最优动作值）。[^src-chapter-7-temporal-difference-methods]

本章的重要统一视角是：这些算法都可写成“当前估计 + 学习率 ×（目标 - 当前估计）”的同构更新形式，差异仅在目标构造方式。由此可把算法比较转化为对目标偏差、方差与稳定性的比较。[^src-chapter-7-temporal-difference-methods]

实践层面，本章强调行为策略与目标策略的分离（off-policy 价值）以及学习率选择对收敛速度和稳定性的影响。它为后续函数逼近与深度 RL 奠定了算法语义基础。[^src-chapter-7-temporal-difference-methods]

[^src-chapter-7-temporal-difference-methods]: [[source-chapter-7-temporal-difference-methods]]
