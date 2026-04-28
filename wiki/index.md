---
title: Index
type: overview
created: 2026-04-26
last_updated: 2026-04-27
tags:
  - meta
---

# Wiki Index

All wiki pages, organized by type. Updated on every ingest.

## Overview
- [[overview]] — wiki scope and current state

## Sources
- [[source-hyperd-hybrid-periodicity-decoupling]] — "HyperD: Hybrid Periodicity Decoupling Framework for Traffic Forecasting" (Wen & Feng, 2025)
- [[source-math-foundation-rl-readme]] — "Mathematical Foundations of Reinforcement Learning" 配套 Readme（Zhao, 2025）
- [[source-grid-world-code-readme]] — 网格世界环境官方代码说明（Zhao, Mi & Li）
- [[source-chapter-1-basic-concepts]] — 第 1 章：强化学习基础概念与 MDP 形式化
- [[source-chapter-2-state-values-and-bellman-equation]] — 第 2 章：状态值、动作值与贝尔曼方程
- [[source-chapter-7-temporal-difference-methods]] — 第 7 章：TD 方法、Sarsa 与 Q-learning
- [[source-chapter-3-optimal-state-values-and-bellman-optimality-equation]] — 第 3 章：最优状态值与贝尔曼最优方程
- [[source-chapter-4-value-iteration-and-policy-iteration]] — 第 4 章：值迭代与策略迭代
- [[source-chapter-5-monte-carlo-methods]] — 第 5 章：蒙特卡洛方法与 ε-greedy

## Entities
- [[hyperd]] — Hybrid Periodicity Decoupling framework for traffic forecasting
- [[math-foundation-of-reinforcement-learning]] — 《Mathematical Foundations of Reinforcement Learning》教材（Springer, 2025）
- [[shiyu-zhao]] — 赵世钰，强化学习教材作者，西湖大学教授
- [[grid-world-environment]] — 网格世界环境，强化学习教学示例

## Concepts
- [[hybrid-periodicity-decoupling]] — explicitly separating short-term and long-term periodicity in time-series signals
- [[traffic-forecasting]] — predicting future traffic states from historical sensor data
- [[mdp-formal-definition]] — 马尔可夫决策过程的六要素与马尔可夫性质
- [[action-value-function]] — 动作值函数 q(s,a) 及其与状态值关系
- [[exploration-vs-exploitation]] — 强化学习中的探索与利用权衡
- [[contraction-mapping-theorem]] — 最优性算子收敛分析的数学基础

## Techniques
- [[frequency-aware-residual-representation]] — Fourier-based signal decomposition into periodic and residual components
- [[spatial-temporal-attentive-encoder]] — dual-pathway encoder for short-term and long-term periodicity
- [[dual-view-alignment-loss]] — regularization loss aligning representations across periodicity views
- [[demlp-decoder]] — two-stage coarse-to-fine decoder with explicit trend removal
- [[bellman-equation]] — 贝尔曼方程，强化学习中值函数的递归关系
- [[temporal-difference-learning]] — 时序差分学习，包括 Sarsa、Q-learning 等算法
- [[policy-evaluation]] — 给定策略下求解状态值函数的过程
- [[sarsa-algorithm]] — on-policy TD 控制算法
- [[expected-sarsa]] — 基于期望目标的 Sarsa 变体
- [[n-step-sarsa]] — 连接 TD 与 MC 的多步 Sarsa 方法
- [[q-learning-algorithm]] — off-policy TD 控制算法
- [[bellman-optimality-equation]] — 最优值函数与最优策略的不动点方程
- [[value-iteration]] — 直接迭代求解最优方程的动态规划算法
- [[policy-iteration]] — 策略评估与改进交替的动态规划算法
- [[truncated-policy-iteration]] — 值迭代与策略迭代之间的截断折中
- [[monte-carlo-methods-rl]] — 基于轨迹回报的无模型策略优化方法
- [[epsilon-greedy]] — 经典探索策略

## Analyses
- [[rl-learning-path-mfrl]] — 基于《Mathematical Foundations of Reinforcement Learning》的系统性 RL 学习路径
- [[on-policy-vs-off-policy]] — on-policy 与 off-policy 学习范式对比
