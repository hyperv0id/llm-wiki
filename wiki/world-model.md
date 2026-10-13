---
title: "World Model"
type: concept
tags:
  - world-model
  - cognitive-architecture
  - energy-based-model
  - uncertainty
created: 2026-10-13
last_updated: 2026-10-13
source_count: 1
confidence: medium
status: active
---

# World Model

在 LeCun 2022 年立场论文中，世界模型是所提架构中最复杂的部件，论文将其定位为未来数十年 AI 进展的主要障碍[^src-autonomous-machine-intelligence]。以内部模型预测状态演化的思想在心理学可追溯至 Craik (1943)，在最优控制中自 1950 年代起即为模型预测控制的标准做法；论文指出，强化学习中长期被忽视的可微世界模型正在回归[^src-autonomous-machine-intelligence]。

## 两个职责

1. 补全感知未提供的世界状态信息；
2. 预测合理的未来世界状态（包括环境自然演化与 actor 提议动作序列导致的后果）。

世界模型由表示不确定性的隐变量参数化，可预测多个可能的世界状态；其作为任务相关环境特性的模拟器，预测在抽象表征空间中进行，并在多个抽象层级与时间尺度上操作世界状态表征[^src-autonomous-machine-intelligence]。

## 三个待解问题

论文列出训练世界模型的三个问题[^src-autonomous-machine-intelligence]：

1. **训练数据的多样性**：取决于 agent 所采取的动作，涉及序列决策中训练集多样性的一般问题。
2. **多模态预测**：世界非完全可预测，给定状态与动作存在多个合理结果；世界模型必须能有效表示这个可能无限的预测集合。
3. **多时间尺度与多抽象层级**：人在抽象层规划复杂目标，再通过短时程预测逐级分解至毫秒级肌肉控制，且由局部条件决定具体实例化。

论文将第 2 个问题视为关键难点与所提方案的核心解法所在，第 3 个问题的解法则是 [[hierarchical-jepa|H-JEPA]][^src-autonomous-machine-intelligence]。

## 不确定性的分类

论文把未来状态预测的不确定性分为两类七种[^src-autonomous-machine-intelligence]：

| 类型 | 来源 |
|------|------|
| aleatoric 1 | 世界本身随机 |
| aleatoric 2 | 世界确定但混沌，感知精度有限时难以预测 |
| aleatoric 3 | 世界确定但部分可观测 |
| epistemic 1 | 世界完全可观测，但传感器只给出部分信息 |
| epistemic 2 | 感知模块提取的状态表征不含准确预测所需的全部信息 |
| epistemic 3 | 世界模型因表征能力有限而不准确（有界理性） |
| epistemic 4 | 世界模型因训练数据有限而不准确 |

强化学习文献常直接要求模型、critic 与策略表示分布；本提案则将预测变量可能的随机性推进隐变量，隐变量可被优化、预测或采样。论文注明这对应机器学习文献中的 reparameterization trick，但在该架构中隐变量参数化被视为根本机制而非单纯技巧[^src-autonomous-machine-intelligence]。

## 预测在表征空间

世界模型不预测像素。论文以视频预测为例：地毯纹理、风中树叶、水面涟漪等细节无法在长时间跨度上准确预测，除非消耗巨大资源；[[jepa|JEPA]] 的优势在于主动忽略不易预测的输入细节，并通过训练判据保证被忽略细节的信息量被压到最低[^src-autonomous-machine-intelligence]。

## 世界状态的跟踪

世界状态不应按常规深度学习架构以向量或多维数组整块传递：状态逐时刻通常仅微小变化（例如把瓶子从厨房搬到餐厅只改动瓶子、厨房、餐厅的状态），整块传递效率低下。因此世界状态应保存在可写记忆中，事件发生时仅更新受影响部分[^src-autonomous-machine-intelligence]。

实现采用键值联想记忆，形式与 memory-augmented networks、entity networks 同类[^src-autonomous-machine-intelligence]：

$$\mathrm{Mem}(q) = \sum_j c_j v_j, \qquad \tilde{c}_j = \mathrm{Match}(k_j, q), \qquad c = \mathrm{Normalize}(\tilde{c})$$

其中 $k_j$ 是键、$v_j$ 是存储值，$\mathrm{Match}(k,q)$ 度量键与查询的差异，$\mathrm{Normalize}$ 做竞争性归一化或阈值化，论文举的例子是 $c_j = \exp(\tilde{c}_j)/[\gamma + \sum_k \exp(\tilde{c}_k)]$，$\gamma$ 为正的常数。写入复用同一套匹配与归一化，再以 $\mathrm{Update}(r, v_j, c_j) = c\,r + (1-c)\,v_j$ 之类的规则更新已有条目；若查询与所有键都相距很远，则分配一个键为 q、值为 r 的新条目，$\gamma$ 可充当键-查询差异的阈值。论文指出每个条目可视为世界中一个实体的状态，并在瓶子例子里给出 $k_{\text{bottle}}$、$k_{\text{kitchen}}$、$k_{\text{dining-room}}$ 三个键及事件前后的值更新；这些操作均可微，梯度可以穿过它们回传[^src-autonomous-machine-intelligence]。

## 模块架构的选择

最佳模块架构依赖环境类型，论文给出两条倾向：低层短期视频预测适合提取局部特征向量、按预测的运动把特征向量从一帧位移到下一帧，隐变量可编码位移图来调制帧间路由连接；高层长期预测中相关特征是物体及其交互，适合用对置换等变的 transformer 捕捉离散物体间的相互作用[^src-autonomous-machine-intelligence]。

论文另提出将世界模型与自我模型解耦：自然世界复杂且不可完全预测，需要带隐变量的高容量模型；agent 自身执行器动作产生的运动往往可以确定性预测，因而应具备独立的、可能不含隐变量的自我模型（论文引用 Sobal et al. 2022）；自我模型亦可作为多智能体场景中其他 agent 模型的模板[^src-autonomous-machine-intelligence]。

## 数据从哪来

论文列出 agent 了解世界的五种信息获取方式：被动观察（被喂传感器流）；主动注视（可在不影响环境的前提下调整注意焦点，如转动视听觉传感器）；被动施动（观察另一个作用于环境的 agent 以推断动作对环境状态的因果影响）；主动自运动（可在几乎不影响环境的前提下改变传感器位置，含可操纵的主动传感器与触觉）；主动施动（传感器流受 agent 动作影响，从而建立可预测自身动作后果的因果模型，并直面探索-利用困境）[^src-autonomous-machine-intelligence]。

论文把"多少能靠被动观察（模式 1、2、4）学到、多少需要自运动（模式 3）、多少需要完整施动性（模式 5）"列为开放问题；并指出要主动收集最大化理解的信息，可能需要内在动机模块驱动注意力、好奇心与探索，去往世界模型预测目前不精确或不确定的状态空间角落[^src-autonomous-machine-intelligence]。

## Related Pages

- [[jepa]] — 论文提议的世界模型架构及其失效证据
- [[hierarchical-jepa]] — 多层级、多时间尺度的世界模型
- [[autonomous-agent-architecture]] — 世界模型在整体架构中的位置
- [[mode-1-mode-2-planning]] — 世界模型在 Mode-2 中的用法

[^src-autonomous-machine-intelligence]: [[source-autonomous-machine-intelligence]]
