---
title: "Grid World Environment Code (Readme)"
type: source-summary
tags:
  - reinforcement-learning
  - grid-world
  - python
  - matlab
  - environment
created: 2026-04-27
last_updated: 2026-04-27
source_count: 1
confidence: high
status: active
---

# Code for the Grid-World Environment — Readme

该文档是《Mathematical Foundations of Reinforcement Learning》书中[[grid-world-environment|网格世界环境]]的官方代码说明，由作者 [[shiyu-zhao|Shiyu Zhao]] 及其博士生 Yize Mi 和 Jianan Li 共同贡献。[^src-grid-world-code-readme]

## 核心内容

该代码库提供了网格世界环境的 Python 和 MATLAB 两种实现，读者可在此环境中开发和测试自己的强化学习算法。官方不提供书中所有算法的完整实现——这些作为线下教学的学生作业。[^src-grid-world-code-readme]

## Python 版本

- 支持 Python 3.7–3.11，依赖 `numpy` 和 `matplotlib`
- 提供默认示例脚本 `examples/example_grid_world.py`
- 可视化功能：蓝色星标表示智能体当前位置，箭头表示策略，绿线表示历史轨迹，黄色格子表示障碍物，蓝色格子表示目标状态，格子上数字表示状态值
- 可自定义参数：环境大小、起始状态、目标状态、障碍物、各类奖励值[^src-grid-world-code-readme]

## MATLAB 版本

- 需要 MATLAB >= R2020a
- 启动 `main.m` 可生成四张图：策略可视化、随机轨迹、确定性轨迹、状态值热力图
- 箭头长度与选择该动作的概率成正比，圆形表示保持不动[^src-grid-world-code-readme]

## 环境交互

通过 `GridWorld` 类创建实例，提供 `reset()`、`step(action)`、`render()` 等标准 RL 接口。策略可设计为确定性或随机性，以矩阵形式表示。[^src-grid-world-code-readme]

[^src-grid-world-code-readme]: [[source-grid-world-code-readme]]
