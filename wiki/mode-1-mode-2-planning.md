---
title: "Mode-1 and Mode-2 Planning"
type: concept
tags:
  - cognitive-architecture
  - model-predictive-control
  - planning
  - energy-based-model
created: 2026-10-13
last_updated: 2026-10-13
source_count: 1
confidence: medium
status: active
---

# Mode-1 and Mode-2 Planning

LeCun 2022 年立场论文将感知-动作循环划分为两种模式，类比 Kahneman 的 System 1 与 System 2：Mode-1 由感知直接生成动作，不进行规划；Mode-2 经由世界模型与代价评估进行推理与规划[^src-autonomous-machine-intelligence]。

## Mode-1：反应式行为

感知模块经编码器输出状态 $s[0]=\mathrm{Enc}(x)$，actor 内的策略模块计算动作 $a[0]=A(s[0])$ 并送往执行器。策略模块由 [[autonomous-agent-architecture|configurator]] 按任务动态配置；除当前状态外，还可读取短期记忆进行联想动作检索[^src-autonomous-machine-intelligence]。

论文指出 Mode-1 的结构限制在于真实物理环境不可微，无法沿 cost ← perception ← world ← action 反向传播梯度；只能借助多个扰动动作试探环境以估计代价关于动作的梯度，采样低效且具潜在危险，对应强化学习中的经典策略梯度方法[^src-autonomous-machine-intelligence]。

在 Mode-1 运行期间，系统可选择性更新世界模型：世界模型单步预测 $s[1]$，待真实观测到达后作为预测监督目标[^src-autonomous-machine-intelligence]。

## Mode-2：用世界模型推理与规划

论文将 Mode-2 规划流程拆解为七步[^src-autonomous-machine-intelligence]：

1. **感知**：提取状态 $s[0]=P(x)$，代价模块计算并存储该状态的即时代价。
2. **动作提议**：actor 生成初始动作序列 $(a[0],\dots,a[T])$。
3. **仿真**：世界模型预测动作序列导致的一到多条可能状态序列 $(s[1],\dots,s[T])$。
4. **评估**：代价模块评估累积总代价 $F(x)=\sum_{t=1}^{T} C(s[t])$。
5. **规划**：actor 优化动作序列以降低代价（如回传代价对动作变量的梯度），收敛至最优序列 $(\check{a}[0],\dots,\check{a}[T])$；必要时迭代第 2–5 步。
6. **执行**：只把优化序列中第一个（或前几个）动作下发至执行器，推进下一轮感知-动作循环。
7. **记忆**：动作执行后，将状态及 intrinsic cost、critic 计算的代价写入短期记忆，用于后续训练与校准 critic。

该流程对应最优控制中的滚动时域模型预测控制（MPC），核心差异在于世界模型与代价函数均由学习习得[^src-autonomous-machine-intelligence]。

优化算法不限于梯度法：当动作-代价映射存在不连续时，可采用动态规划、组合优化、模拟退火、无梯度方法或带剪枝的启发式树搜索；若动作空间小且可离散化，动态规划尤为适用[^src-autonomous-machine-intelligence]。

## 不确定性的来源

论文将单步状态转移的不确定性归为三类：物理环境本身的随机性（aleatoric）、状态表征 $s[t]$ 因观测不完全导致的信息缺失（epistemic），以及世界模型受训练数据、表征容量或算力限制带来的模型误差[^src-autonomous-machine-intelligence]。分类细节见 [[world-model]]。

## 从 Mode-2 到 Mode-1：把技能编译成策略

Mode-2 的计算开销较大：系统仅配备单个由 configurator 动态调配的世界模型引擎，一次只能专注单项复杂任务；Mode-1 仅需单次策略前向计算，且可同时维持多个面向不同任务的策略模块[^src-autonomous-machine-intelligence]。

论文提出的技能编译流程（Figure 5）基于摊销推断（amortized inference）：先在 Mode-2 下规划得出最优动作序列 $(\check{a}[0],\dots,\check{a}[T])$，随后训练策略模块参数，最小化其输出与最优动作的散度 $D(\check{a}[t], A(s[t]))$。训练完成后，策略模块既可在 Mode-1 下直接产出动作，将复杂规划“编译”为反应式策略；亦可在 Mode-2 优化前递归提供初始动作序列[^src-autonomous-machine-intelligence]。

## 推理即能量最小化

Mode-2 中搜索动作序列的过程在论文里被定义为一种推理：以世界模型前向仿真为基础，对动作序列执行能量最小化优化；广义上，“动作”可泛化为表示状态间抽象转换的隐变量。论文把基于仿真与优化的规划列为自然智能中最常见的推理形式[^src-autonomous-machine-intelligence]。

在形式化层面，该架构对应因子图，代价模块充当 log 因子；经典 AI 中图模型的概率推断亦可表述为优化或约束满足问题。论文同时指出该架构支持传统逻辑与概率推断之外的推理范式，即基于仿真与类比的推理[^src-autonomous-machine-intelligence]。

论文在 8.3.3 节讨论离散符号与抽象决策需求：高层抽象决策多具定性特征（例如自动驾驶高层决策为路口转向选择，低层对应转向角连续控制序列），动作空间本质离散，可通过穷举搜索、启发式搜索（含 MCTS）或满足 Bellman 方程时的动态规划求解[^src-autonomous-machine-intelligence]。

## Related Pages

- [[autonomous-agent-architecture]] — actor 与其余模块的连接
- [[intrinsic-cost-and-critic]] — Mode-2 中被最小化的代价如何构成
- [[hierarchical-jepa]] — 把 Mode-2 扩展到多层级、多时间尺度
- [[world-model]] — 仿真步骤所依赖的世界模型

[^src-autonomous-machine-intelligence]: [[source-autonomous-machine-intelligence]]
