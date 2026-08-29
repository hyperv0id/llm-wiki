---
title: "PyPOTS"
type: entity
tags:
  - toolbox
  - data-imputation
  - time-series
  - partially-observed-time-series
  - benchmark
created: 2026-08-29
last_updated: 2026-08-29
source_count: 1
confidence: medium
status: active
---

# PyPOTS

**PyPOTS** 是 Du 于 KDD 2023 workshop 提出的 Python 工具箱，专注于对部分观测时间序列（partially-observed time series）进行端到端建模；据 MTSI 综述统计，截至综述写作时其包含 37 个插补模型，覆盖不完整时间序列上的多种任务[^src-mts-imputation-survey]。

综述在梳理 MTSI 工具箱时重点介绍 **PyPOTS Ecosystem**：它整合了多样插补算法、标准化流水线与基准资源，作者将其定位为促进可及、可复现 MTSI 研究的基础设施[^src-mts-imputation-survey]。基于该生态，同一团队构建了 **TSI-Bench** 基准套件：为 172 个公开时间序列数据集提供标准化插补基准流水线；综述报告其基准结果来自 34,804 组实验，覆盖 28 个算法与 8 个典型领域数据集及多种缺失模式[^src-mts-imputation-survey]。

## 综述列出的其他插补工具箱

综述对深度学习之外的插补工具也做了梳理（综述口径）[^src-mts-imputation-survey]：

- **imputeTS**（R）：均值、末次观测结转等朴素方法与线性插值、Kalman 平滑等，仅支持单变量序列
- **mice**（R）：链式方程多重插补，非时序专用但统计领域广泛用于多元时序
- **Impyute / Autoimpute**：横截面与时序数据的朴素插补（移动平均、多项式/样条插值等）
- **GluonTS**：生成式时序机器学习包，提供 dummy value、causal mean 等朴素缺失处理
- **Sktime**：可调用机器学习插补算法，但仅单变量
- **ImputeBench**：收集机器学习与深度学习插补方法，但缺乏统一编程语言

## 关联页面

- [[mts-imputation-taxonomy]] — 综述的分类框架，PyPOTS 生态是综述三大贡献之一
- [[source-mts-imputation-survey]] — 来源综述摘要
- [[csdi]] / [[grin]] / [[imputeformer]] — PyPOTS 所收录算法类型的方法实例
- [[tslib]] — 时序预测方向的另一个开源库（对照）

[^src-mts-imputation-survey]: [[source-mts-imputation-survey]]
