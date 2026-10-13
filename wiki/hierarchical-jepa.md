---
title: "Hierarchical JEPA (H-JEPA)"
type: technique
tags:
  - self-supervised-learning
  - world-model
  - hierarchical-planning
  - energy-based-model
created: 2026-10-13
last_updated: 2026-10-13
source_count: 1
confidence: medium
status: active
---

# Hierarchical JEPA (H-JEPA)

LeCun 2022 年立场论文将 [[jepa|JEPA]] 的层级堆叠（H-JEPA）定位为非生成式层级预测世界模型：用于学习多层级、多时间尺度的表征，并作为底座支持不确定性下的分层规划（Mode-2）[^src-autonomous-machine-intelligence]。

## 动机

低层表征保留输入细节，仅适于短期预测；高层抽象表征滤除无关细节，适于长时程预测。论文以驾驶为例：根据未来数秒的转向与踏板序列可精确预测车辆轨迹；长时程下的轨迹细节受外部事件影响而不可预测，但司机可在高层抽象上预测大致到达时间与宏观路径，备选路线由离散隐变量表示[^src-autonomous-machine-intelligence]。

论文反对生成式架构的理由是：生成式隐变量模型无法消除不可预测的无关细节，只能将其推入隐变量，无法产生 y 的抽象（不变）表征[^src-autonomous-machine-intelligence]。

## 机制

Figure 15 展示其堆叠结构：JEPA-1 从输入 $x_0,x_1,x_2$ 提取低层表征进行短期预测；JEPA-2 接收 JEPA-1 的表征，提取更高层表征进行更长时程预测。架构可扩展至更多层，层间通过 temporal pooling 粗粒度化表征；训练可按层进行，也可全局进行，采用任意非对比方法[^src-autonomous-machine-intelligence]。

## 分层规划

Figure 16 给出 Mode-2 分层规划架构。感知系统经级联编码器生成多级表征 $s[0]=\mathrm{Enc}_1(x)$、$s_2[0]=\mathrm{Enc}_2(s[0]),\dots$；各层均执行预测，高层负责长时程，低层负责短时程[^src-autonomous-machine-intelligence]。

顶层推断高层"动作"序列 $(a_2[2],a_2[4])$ 以最小化高层目标 $C(s_2[4])$；高层动作并非实际物理动作，而是低层状态需满足的约束条件。代价模块 $C(s[2])$、$C(s[4])$ 接收低层状态 $s[2]$ 与高层条件 $a_2[2]$ 并度量满足程度，为低层设定子目标；低层进而推断使子目标代价最小的动作序列。论文指出该流程为自顶向下的贪心推断，理想方案是对各层动作序列进行联合优化[^src-autonomous-machine-intelligence]。

论文将"动作仅是下层需满足的条件"溯源至控制论：经典比例伺服机构中，二次代价度量目标状态与当前状态的平方距离，控制量正比于代价对动作变量的负梯度[^src-autonomous-machine-intelligence]。

## 不确定性下的分层规划

Figure 17 在各层、各时间步的预测器中引入隐变量。预测 $s_2[2]$ 需采样隐变量 $z_2[2]$，其分布负对数对应正则器 $R_2(z_2[2])$。正则器参数可设为常数（如固定高斯）、由当前数据摊销推断（如由 $s_2[0]$ 计算多项式或高斯参数），或由 [[autonomous-agent-architecture|configurator]] 配置；论文指出利用先前预测配置正则器可使系统偏向生成"好"的轨迹[^src-autonomous-machine-intelligence]。

轨迹空间随步数指数增长：若每个隐变量有 k 个离散取值，t 步后存在 $k^t$ 条可能轨迹，需依赖定向搜索与剪枝（如经典 MCTS）；连续隐变量则从正则器定义的连续分布采样。给定隐变量样本后可推断各层最优动作序列，但需多次重采样以覆盖合理结果；该推理过程不仅最小化期望代价，同时最小化期望代价的方差以抑制不确定性[^src-autonomous-machine-intelligence]。

## 未验证项

8.1 节列出的开放问题：从视频构建并训练 H-JEPA 以及能否学出 4.1 节设想的抽象概念层级为首要开放问题；隐变量正则化的最佳方案未定；[[autonomous-agent-architecture|configurator]] 如何学习将复杂任务分解为子目标序列被明确留待未来研究[^src-autonomous-machine-intelligence]。本页所述机制均为论文架构提议，尚无配套实验。

## Related Pages

- [[jepa]] — 单层 JEPA 的能量形式与训练判据
- [[autonomous-agent-architecture]] — 层级世界模型在整体架构中的位置
- [[mode-1-mode-2-planning]] — Mode-2 的非层级版本与推理即能量最小化

[^src-autonomous-machine-intelligence]: [[source-autonomous-machine-intelligence]]
