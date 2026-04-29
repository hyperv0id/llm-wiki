---
title: "Chapter 1: Basic Concepts"
type: source-summary
tags:
  - reinforcement-learning
  - mdp
  - grid-world
  - textbook
created: 2026-04-27
last_updated: 2026-04-27
source_count: 1
confidence: medium
status: active
---

# Chapter 1: Basic Concepts

本章构建了本书的统一问题表述：用网格世界解释强化学习中的状态、动作、策略、奖励与回报，并给出马尔可夫决策过程（MDP）的正式定义。[^src-chapter-1-basic-concepts]

章节先以 5×5 网格世界为主例，说明动作集合可包含上下左右与停留，状态转移由 $p(s'|s,a)$ 建模；边界与禁区会改变可达性与奖励分布。该设定用于后续全部算法章节，起到“统一实验坐标系”的作用。[^src-chapter-1-basic-concepts]

在概念层面，本章强调“即时奖励不等于长期目标”：策略优劣由累积回报（return）决定，而非单步奖励。对无限时间问题，引入折扣因子 $\gamma\in(0,1)$，使回报有限并可调节短视/远视偏好。[^src-chapter-1-basic-concepts]

在任务类型上，本章区分 episodic 与 continuing，并用 absorbing state（吸收态）把终止问题转成统一的马尔可夫框架，便于后续推导贝尔曼方程与 TD 更新。[^src-chapter-1-basic-concepts]

最后，本章给出有限 MDP 的核心组件与马尔可夫性质：下一状态与奖励只依赖当前状态和动作。并指出固定策略后，MDP 退化为马尔可夫过程（MP），从而把“控制”问题分解为“评估 + 改进”的结构化流程。[^src-chapter-1-basic-concepts]

局限是：本章以有限离散模型为主，连续状态/动作与函数逼近问题留到后续章节处理。[^src-chapter-1-basic-concepts]

[^src-chapter-1-basic-concepts]: [[source-chapter-1-basic-concepts]]
