---
title: "Intrinsic Cost and Trainable Critic"
type: concept
tags:
  - cognitive-architecture
  - intrinsic-motivation
  - energy-based-model
  - reward
created: 2026-10-13
last_updated: 2026-10-13
source_count: 1
confidence: medium
status: active
---

# Intrinsic Cost and Trainable Critic

LeCun 2022 年立场论文将 cost 模块划分为两个互补组件。该模块输出单一标量能量度量 agent 的不适程度；agent 的总体目标是通过动作优化，使系统维持在平均能量最小的状态[^src-autonomous-machine-intelligence]。

## 形式

总 cost 由 intrinsic cost 与 trainable critic 线性组合构成[^src-autonomous-machine-intelligence]：

$$C(s) = IC(s) + TC(s), \qquad IC(s) = \sum_{i=1}^{k} u_i\, IC_i(s), \qquad TC(s) = \sum_{j=1}^{l} v_j\, TC_j(s)$$

各子模块对应特定的行为驱力，权重 $u_i$ 与 $v_j$ 由 [[autonomous-agent-architecture|configurator]] 动态调节，以引导 agent 在不同阶段聚焦不同子目标[^src-autonomous-machine-intelligence]。

## Intrinsic Cost（不可训练）

IC 属于硬连线模块，不可训练且不可被外部修改。其输入为感知模块输出的当前状态或世界模型预测的状态，计算瞬时能量，决定 agent 的基本驱力与内在动机边界[^src-autonomous-machine-intelligence]。

论文列举了三类 IC 驱力：(1) 生理与安全边界，度量极端受力、危险环境（电气/化学/热）、功耗过高或电量告急等状态，类比疼痛与恐惧；(2) 技能与社交先验，如驱动腿式机器人站立行走，或寻求人类互动与赞许并回避人类痛苦；(3) 探索驱力，例如好奇心或倾向于产生可观测影响的动作，用以提升世界模型训练数据的多样性[^src-autonomous-machine-intelligence]。

IC 设为不可训练的目的是防止行为策略塌缩或向有害行为失控漂移；论文将其功能类比为哺乳动物大脑中的杏仁核[^src-autonomous-machine-intelligence]。

## Trainable Critic

TC 承担两项功能：在尽量少调用世界模型的前提下预判长期代价，以及供 configurator 引入学得的代价以引导 agent 完成子目标[^src-autonomous-machine-intelligence]。

其训练机制如图 Figure 7 所示：规划期间 IC 将三元组 $(\tau, s_\tau, IC(s_\tau))$ 写入短期记忆；训练 critic 时提取历史状态 $s_\tau$ 与后续内在能量 $IC(s_{\tau+\delta})$，通过回归损失（如 $\|IC(s_{\tau+\delta}) - TC(s_\tau)\|^2$）最小化预测误差，亦可推广至预测折扣未来能量的期望或分布[^src-autonomous-machine-intelligence]。状态向量可包含动作信息，且采样状态既可来自真实感知，亦可来自 Mode-2 中世界模型的想象轨迹；论文指出该过程在机制层面类似于 A2C 等强化学习算法中的 critic 训练[^src-autonomous-machine-intelligence]。

## 行为规定的四种方式

论文总结了规范 agent 行为的四种方式：(1) 显式硬编码特定条件下的触发行为；(2) 设计目标函数，使行为通过最小化代价规划自动涌现；(3) 监督学习训练 Mode-1 策略以复现专家动作；(4) 逆强化学习，即通过观察专家反推其隐含的目标函数并构建 Mode-2 所需的 critic[^src-autonomous-machine-intelligence]。第二种方式比第一种更易工程化也更稳健：硬编码规则易因未预期环境变化而失效，目标函数驱动则让 agent 自适应调整动作序列，同时减少对脆弱人工先验的依赖[^src-autonomous-machine-intelligence]。

## 与强化学习的关系

多数强化学习设定将 reward（或负 reward 形式的 cost）交由环境提供，即 IC 本质上属于未知环境函数，梯度需经由动作试错估计（如策略梯度），或依赖 actor-critic 拟合可微近似[^src-autonomous-machine-intelligence]。在该架构中，核心学习过程集中于世界模型，梯度可直接经由模型反向传播，reward 在驱动学习中的地位相对次要；论文在此明确反对 "reward is enough" 假说[^src-autonomous-machine-intelligence]。

## Related Pages

- [[autonomous-agent-architecture]] — 代价模块在整体架构中的位置与 configurator 的调制
- [[mode-1-mode-2-planning]] — 代价如何驱动 Mode-2 的动作序列搜索
- [[energy-based-model]] — 能量作为标量目标函数的一般背景

[^src-autonomous-machine-intelligence]: [[source-autonomous-machine-intelligence]]
