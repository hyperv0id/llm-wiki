---
title: "PAST"
type: entity
tags:
  - traffic-imputation
  - spatio-temporal
  - graph-neural-network
  - pvldb
  - missing-data
  - primary-auxiliary-patterns
created: 2026-06-15
last_updated: 2026-06-15
source_count: 1
confidence: medium
status: active
---

# PAST (Primary-Auxiliary Spatio-Temporal Network)

PAST 是上海交通大学提出的交通时间序列填补模型，发表于 PVLDB（arXiv: 2511.13414, 2025），代码开源在 github.com/Hanwen-Hu/PAST[^src-past]。核心创新是将时空模式划分为**主模式（primary patterns）**和**辅助模式（auxiliary patterns）**，通过双模块架构分别建模。

## 动机

交通时间序列面临三种缺失类型：随机缺失（Random）、纤维缺失（Fiber，传感器长时间离线）、块缺失（Block，大范围传感器缺失）。现有模型要么仅处理随机缺失，要么将多种模式纠缠建模，未建立缺失类型与模式的正交对应[^src-past]。

PAST 的核心洞察：随机缺失可通过局部数据关系（主模式）解决，而纤维和块缺失需要外部信息（辅助模式）来捕获长程周期性和大范围空间相似性——例如早高峰源自固定出行时间安排，而非前一天的早高峰。

## 架构

PAST 由两个交互模块组成：

- **GIM (Graph-Integrated Module)**：纯 GNN，通过动态有向图 + [[interval-aware-dropout]] + 多阶卷积捕获主模式
- **CGM (Cross-Gated Module)**：通过 [[cross-gated-mechanism|交叉门控机制]] 从时间戳和节点属性等外部特征提取辅助模式
- 两层间通过共享隐向量交换信息，采用受 GBDT 启发的 ensemble 训练框架

最终填补：$\boldsymbol{Y} = M \odot X + (1-M) \odot (Y_{CGM} + Y_{GIM})$

## 性能

在 METR-LA、PeMS-Bay、LargeST-SD 上 27 种缺失条件下 vs 7 个基线：
- 随机缺失：离线 RMSE 降低 ~8.2%
- 纤维缺失：离线 RMSE 降低 ~12.8%
- 块缺失：在线 RMSE 降低 **26.2%**、MAE 降低 **31.6%**

在纤维和块缺失场景下优势显著，验证了辅助模式对结构化缺失的支撑作用。

## 与其他模型的对比

| 模型 | 缺失类型覆盖 | 模式策略 | 外部信息 |
|------|-------------|---------|---------|
| PAST | 随机+纤维+块 | Primary-Auxiliary 解耦 | CGM 专用模块 |
| [[grin\|GRIN]] | 随机+纤维 | 内部主模式统一建模 | 无 |
| [[csdi\|CSDI]] | 随机 | 扩散条件去噪 | 无 |
| [[stcpa\|STCPA]] | 随机+纤维+块 | 内部主模式 | 无 |
| [[t1\|T1]] | 随机+纤维+块 | CD+CI 混合 | 无 |

## 相关页面

- [[primary-auxiliary-patterns]] — 主-辅模式的详细定义
- [[interval-aware-dropout]] — GIM 中的时序 dropout 机制
- [[cross-gated-mechanism]] — CGM 中的交叉门控操作
- [[grin]], [[csdi]], [[cofill]], [[sadi]], [[fence]], [[t1]] — 其他填补模型
- [[missing-not-at-random]] — 缺失机制分类

[^src-past]: [[source-past]]
