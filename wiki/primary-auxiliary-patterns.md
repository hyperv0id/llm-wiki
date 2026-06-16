---
title: "Primary-Auxiliary Patterns"
type: concept
tags:
  - spatio-temporal
  - missing-data
  - pattern-disentanglement
  - traffic-imputation
created: 2026-06-15
last_updated: 2026-06-15
source_count: 1
confidence: medium
status: active
---

# Primary-Auxiliary Patterns (主-辅模式)

Primary-Auxiliary Patterns 是 [[past|PAST]] 引入的时空模式分类框架，用于解决交通时间序列填补中不同缺失类型需要不同建模策略的问题[^src-past]。

## 定义

### 主模式 (Primary Patterns)

源自时间序列内部数据点之间的关系——不依赖任何外部信息即可从观测值推断的模式：

- 局部时序波动：当前值通常可从相邻前值估计
- 邻域拓扑依赖：上游节点的变化直接影响下游节点
- 在随机缺失场景下占主导地位

### 辅助模式 (Auxiliary Patterns)

源自外部因素的模式，用于高效表征长期周期性和大范围空间相似性：

- 长期周期性：早高峰源于固定出行时间安排（而非前一天的早高峰）
- 空间相似性：节点的独特波动模式源自其局部环境上下文（而非直接节点交互）
- 在纤维和块缺失场景下成为关键支撑

## 缺失类型对应

```
随机缺失 (Random Missing)  → 主模式主导，相邻观测值即可推断
纤维缺失 (Fiber Missing)    → 辅助模式必需，长程周期性起关键作用
块缺失 (Block Missing)      → 辅助模式必需，大范围空间相似性弥补
```

这一对应关系是 PAST 方法论的核心：通过为主模式和辅助模式分别设计专用模块，模型可以在不同缺失场景下自适应地利用最有效的模式类型[^src-past]。

## 与传统模式的对比

| 维度 | 传统空间/时间解耦 | Primary-Auxiliary 解耦 |
|------|-------------------|----------------------|
| 划分标准 | 维度来源（空间 vs 时间） | 信息源（内部 vs 外部） |
| 周期性的归属 | 属于时序维度 | 属于辅助模式（外部驱动） |
| 对缺失类型的适应性 | 无显式对应 | 自然对应随机/纤维/块三类缺失 |
| 代表模型 | [[grin\|GRIN]], [[dst-mamba\|DST-Mamba]] | [[past\|PAST]] |

传统模型通常沿空间-时间维度解耦模式，但周期性（如早晚高峰）本质上由外部因素驱动，归入时序维度缺乏因果支撑。Primary-Auxiliary 框架从信息源角度重新划分，使模式提取与具体缺失场景对齐[^src-past]。

## 模型中的实现

在 [[past|PAST]] 中：
- **GIM** 负责主模式提取（时间有向图 + interval-aware dropout + 多阶空间图卷积）
- **CGM** 负责辅助模式提取（时间戳/节点属性嵌入 + cross-gated 交互）
- 两模块通过共享隐向量和 ensemble 残差训练（GBDT 启发）协同工作

## 相关页面

- [[past]] — PAST 模型实体
- [[interval-aware-dropout]] — GIM 中的 dropout 策略
- [[cross-gated-mechanism]] — CGM 中的门控机制
- [[missing-not-at-random]] — 缺失机制分类
- [[spatio-temporal-decoupling]] — 另一种解耦范式

[^src-past]: [[source-past]]
