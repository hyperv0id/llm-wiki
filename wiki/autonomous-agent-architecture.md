---
title: "Autonomous Agent Architecture (LeCun 2022)"
type: concept
tags:
  - cognitive-architecture
  - world-model
  - intrinsic-motivation
  - energy-based-model
  - autonomous-agent
created: 2026-10-13
last_updated: 2026-10-13
source_count: 1
confidence: medium
status: active
---

# Autonomous Agent Architecture (LeCun 2022)

LeCun 2022 年立场论文提出可配置认知架构（Figure 2），其四项贡献之一是"所有模块可微、多数可训练的整体认知架构"[^src-autonomous-machine-intelligence]。模块全可微意味着任一模块均可接收下游代价标量对自己输出的梯度估计[^src-autonomous-machine-intelligence]。

## 模块

| 模块 | 职责 |
|------|------|
| configurator | 执行控制。接收所有其他模块的输入，为当前任务预配置感知、世界模型、代价与 actor，方式是调制它们的参数与注意力回路 |
| perception | 从传感器信号估计当前世界状态；可分层表征；对给定任务只有一小部分状态相关 |
| world model | 补全感知未提供的信息；预测合理的未来状态（自然演化，或 actor 提议动作序列的后果） |
| cost | 输出单一标量"能量"，度量 agent 的不适程度；由不可训练的 intrinsic cost 与可训练的 critic 相加而成 |
| short-term memory | 保存过去、当前与未来状态及对应 intrinsic cost；世界模型与 critic 读写它 |
| actor | 提出动作序列；用世界模型与代价评估；输出最优序列的第一个动作 |

世界模型为架构中最复杂的部分；代价模块见 [[intrinsic-cost-and-critic]]，actor 的两种工作模式见 [[mode-1-mode-2-planning]]，世界模型的职责与不确定性处理见 [[world-model]]。

## 为什么需要 configurator

论文给出两条理由：硬件复用与知识共享。硬件复用使同一套电路可顺序用于多个任务，适应参数内存约束；知识共享则允许为一个环境训练的通用世界模型仅通过 configurator 调制一小部分参数，即可复用于一系列不同任务，避免为每个技能维护一个独立世界模型，节省数据与算力。代价是 agent 一次只能完成一件复杂任务[^src-autonomous-machine-intelligence]。

配置机制包括调制参数与连接图（路由信号、激活子网络、聚焦注意力）。在低层视网膜拓扑特征阵列上采用局部门控与路由；在感知高层与预测器中，若采用 transformer 块实现，configurator 的输出可作为额外输入 token 调制其连接图与功能[^src-autonomous-machine-intelligence]。

论文指出 transformer 块适配该配置需求的两个性质：其函数对置换等变，无需指定物体到特定输入 token 的分配；其功能支持通过增加额外输入 token 配置，从而指定大范围的输入-输出函数[^src-autonomous-machine-intelligence]。

## 代价模块的配置

设置子目标与配置代价模块是 configurator 的核心功能。一种简单做法是调制各基本代价子模块线性组合的权重。论文区分了 intrinsic cost 与 critic 的调制策略：对不可变的 intrinsic cost 做复杂调制可能导致安全护栏等基本驱力失控；critic 则可设计更精巧的架构予以灵活调制。若高层代价定义为物体间期望关系的集合（"螺母是否拧在螺丝上"），可训练 transformer 度量世界状态偏离该条件的程度，并同样借助额外输入 token 调制[^src-autonomous-machine-intelligence]。

## 与脑结构的对应

该对应在原文中标注为推测性，目的是连接认知科学与神经科学概念：感知对应视觉、听觉等感觉皮层及部分联合皮层；世界模型与 critic 对应前额叶皮层不同部位；intrinsic cost 对应基底节中与奖赏相关的结构（包括杏仁核）；短期记忆功能与海马重叠；configurator 对应前额叶中执行控制与注意力调制的结构；actor 对应前运动皮层中编制与编码运动计划的区域[^src-autonomous-machine-intelligence]。

论文以"单一可配置世界模型引擎"作为假说，解释人为什么一次只能进行一件"有意识"的推理与规划任务；并推测意识的错觉可能是大脑中类似 configurator 的模块的副作用[^src-autonomous-machine-intelligence]。

## 未解决

论文在 8.1 节把 configurator 列为全篇最不理解的部件，尤其是它如何在规划复杂任务时识别子目标序列并逐次配置 agent；此外短期记忆的精确架构与功能、模块微架构（预测器所需的动态路由与门控电路）均未指定[^src-autonomous-machine-intelligence]。

## Related Pages

- [[intrinsic-cost-and-critic]] — 代价模块的两个子模块与 critic 训练
- [[mode-1-mode-2-planning]] — actor 的反应式与规划式两条路径
- [[world-model]] — 世界模型的职责、不确定性与数据来源
- [[hierarchical-jepa]] — 层级世界模型与分层规划

[^src-autonomous-machine-intelligence]: [[source-autonomous-machine-intelligence]]
