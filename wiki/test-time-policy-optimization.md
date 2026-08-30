---
title: "Test-Time Policy Optimization (测试时策略优化)"
type: concept
tags:
  - llm-agents
  - reinforcement-learning
  - test-time-learning
  - continual-learning
created: 2026-08-30
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# Test-Time Policy Optimization（测试时策略优化）

Test-time policy optimization 指在不更新模型参数的前提下，于部署/测试阶段持续改进 LLM agent 策略的方法类。JitRL 论文以这一术语定位自身：把冻结的先验策略 $\pi_\theta$ 调制向最优后验 $\pi^*$，以获得持续自进化能力而回避梯度训练的开销与遗忘[^src-jitrl]。

## 问题背景

LLM agent 部署后权重冻结，在陌生或动态环境中无法从交互中学习。论文按"改进信号放在哪里"归纳出三条路线[^src-jitrl]：

1. **梯度 RL 持续更新**（PPO/GRPO、WebRL 等）：以奖励驱动的策略梯度更新权重。通用性强，但训练数据与算力开销大、模型静态化、易灾难性遗忘，且低数据在线场景收敛不稳。
2. **ICL/上下文工程**（Reflexion、AWM、MemGPT/Voyager 类记忆框架）：把历史经验以文本形式放回上下文。免训练，但受上下文长度约束（长交互序列下失效），且 prompt 只能承载能被显式文字描述的知识，缺乏 RL 的奖励驱动通用性。参见 [[in-context-learning]]。
3. **测试时策略优化**（JitRL）：保留 RL 的目标形式（价值、优势、KL 约束），把改进作用在输出分布而非参数上——记忆承载经验，logits 承载更新（见 [[non-parametric-policy-memory]]）。

## 与相邻概念的区别

- 与 [[in-context-learning]]：ICL 的改进信号停留在 prompt 文本层；test-time policy optimization 直接调制输出分布（logits），并以奖励信号而非示例模仿驱动。
- 与权重更新 RL：改进目标同为策略改进（JitRL 的闭式解对应 [[kl-regularized-policy-optimization|KL 正则化策略优化]]，与 WebRL 的目标同形），但作用对象是 logits 而非 θ，因此无遗忘、无反向传播开销，代价是只能在候选动作集内重加权。
- 与时空领域的 [[test-time-adaptation-st]] / [[test-time-computing-st]]：同属"推理时适配"家族，但适配对象是策略分布而非特征统计量或计算预算。

## 代表方法

论文实验覆盖的代表：Reflexion（语言化自我反思）、AWM（工作流记忆）、EvoTest（演化式系统级配置优化）、Memory（FIFO 全量转录），以及梯度侧的 WebRL 与 GRPO；JitRL 在 WebArena/Jericho 上报告全面领先 training-free 基线，并在特定设定下逼近或超过权重更新方法（详见 [[jitrl]]）[^src-jitrl]。

## 相关页面

- [[jitrl]] — 该类的代表方法
- [[non-parametric-policy-memory]] / [[kl-regularized-policy-optimization]] / [[reflective-stepwise-reward]] — 支撑概念
- [[continual-spatio-temporal-forecasting]] — 时空领域的部署后持续学习问题
- [[source-jitrl]] — 源摘要

[^src-jitrl]: [[source-jitrl]]
