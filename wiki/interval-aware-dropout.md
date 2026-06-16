---
title: "Interval-Aware Dropout"
type: technique
tags:
  - dropout
  - time-series
  - graph-neural-network
  - regularization
  - traffic-imputation
created: 2026-06-15
last_updated: 2026-06-15
source_count: 1
confidence: medium
status: active
---

# Interval-Aware Dropout

Interval-Aware Dropout 是 [[past|PAST]] 的 GIM（Graph-Integrated Module）中提出的一种图边 dropout 策略，用于解决时间序列填补中因缺失位置随机变化导致的过拟合问题[^src-past]。

## 动机

在交通时间序列的有向时间图 $G_T$ 中，观测值到缺失值之间存在单向边。传统均匀 dropout 会同等概率丢弃所有边。但交通时间序列的一个经验观察是：**相邻数据点之间的依赖更强，但这种强依赖在纤维/块缺失场景下会失效**——如果缺失点主要依赖相邻点，而当相邻点也缺失时，模型将无法提供准确填补[^src-past]。

## 机制

Interval-Aware Dropout 根据时间距离调整边丢弃概率——越近的边越容易被丢弃：

$$p(\Delta t) = e^{-\alpha \Delta t + \beta}$$

其中：
- $\Delta t$ 为观测点与缺失点的时间距离
- $\alpha$ 控制衰减速度
- $\beta$ 由 $\alpha$ 和期望 dropout 率 $p$ 共同决定：$\beta = \log\frac{pL^2}{\sum\sum e^{-\alpha |j-i|}}$

结果：近邻观测值到缺失值的边以更高概率被丢弃，迫使模型学习跨越更大时间间隔的依赖模式，减少局部过拟合。

## 与标准 Dropout 的对比

| 维度 | 标准 Dropout | Interval-Aware Dropout |
|------|-------------|----------------------|
| 作用对象 | 神经元/节点 | 时间图上的边 |
| 丢弃概率 | 均匀分布 | 时间距离的函数（近高远低） |
| 设计动机 | 防止神经元共适应 | 减少对局部依赖的过拟合 |
| 理论类比 | 正则化防止过拟合 | 同源——选择性边删除增强鲁棒性 |

PAST 中的实现：仅对观测值→缺失值的单向边应用，双向观测值之间的边不受影响。平均 dropout 率设为 $p=0.1$。

## PAST 中的消融验证

消融实验表明，interval-aware dropout 对 PAST 在纤维和块缺失场景下的性能有显著贡献——移除后模型在结构化缺失条件下精度下降，验证了"迫使模型学习更广范围模式"的假设[^src-past]。

## 相关页面

- [[past]] — PAST 模型
- [[cross-gated-mechanism]] — CGM 中的门控机制
- [[primary-auxiliary-patterns]] — 主-辅模式概念

[^src-past]: [[source-past]]
