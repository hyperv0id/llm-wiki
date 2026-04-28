---
title: "Grid World Environment"
type: entity
tags:
  - reinforcement-learning
  - environment
  - python
  - matlab
created: 2026-04-27
last_updated: 2026-04-27
source_count: 3
confidence: high
status: active
---

# Grid World Environment

网格世界（Grid World）是《[[math-foundation-of-reinforcement-learning|Mathematical Foundations of Reinforcement Learning]]》一书中使用的核心示例环境，用于直观地说明强化学习中的概念和算法。[^src-math-foundation-rl-readme]

## 环境描述

网格世界是一个二维网格，智能体在其中移动。环境包含以下元素：
- **智能体**：由蓝色星标表示，可在网格中移动
- **目标状态**：蓝色格子，智能体到达后获得正奖励
- **障碍物**：黄色格子，智能体进入后获得负奖励
- **策略**：每个格子上的箭头表示该状态下的策略方向
- **状态值**：格子上显示的数字，表示该状态的价值[^src-grid-world-code-readme]

在本书第 1 章中，该环境被明确用于统一讲解状态、动作、策略、奖励与回报，并采用边界/禁区/目标的差异化奖励设定（如边界与禁区为负奖励，目标为正奖励）。[^src-chapter-1-basic-concepts]

## 官方代码

[[shiyu-zhao|Shiyu Zhao]] 及其博士生 Yize Mi、Jianan Li 提供了网格世界环境的官方代码，包含 Python 和 MATLAB 两种版本。[^src-grid-world-code-readme]

### Python 版本

- 支持 Python 3.7–3.11，依赖 `numpy` 和 `matplotlib`
- 通过 `GridWorld` 类提供标准 RL 接口：`reset()`、`step(action)`、`render()`
- 可自定义参数：环境大小、起始状态、目标状态、障碍物位置、各类奖励值
- 策略以矩阵形式表示，可设计为确定性或随机性[^src-grid-world-code-readme]

### MATLAB 版本

- 需要 MATLAB >= R2020a
- 启动 `main.m` 可生成策略可视化、轨迹图和状态值图
- 箭头长度与选择该动作的概率成正比，圆形表示保持不动[^src-grid-world-code-readme]

## 教学用途

该环境专为教学目的设计。官方不提供所有算法的完整实现——学生需自行使用该环境完成算法作业。[^src-math-foundation-rl-readme]

[^src-math-foundation-rl-readme]: [[source-math-foundation-rl-readme]]
[^src-grid-world-code-readme]: [[source-grid-world-code-readme]]
[^src-chapter-1-basic-concepts]: [[source-chapter-1-basic-concepts]]
