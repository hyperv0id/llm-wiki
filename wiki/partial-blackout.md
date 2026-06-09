---
title: "Partial Blackout"
type: concept
tags:
  - time-series
  - data-imputation
  - missing-data
  - diffusion-models
created: 2026-06-08
last_updated: 2026-06-09
source_count: 2
confidence: high
status: active
---

# Partial Blackout (部分停电)

**Partial blackout** 是由 [[sadi|SADI]] (AAAI 2025) 引入的更通用的多元时间序列缺失模式[^src-sadi]。它描述了一组特征在连续时间步上同时缺失的情况，并将此前文献中的各种缺失模式统一为特例[^src-sadi]。

## 定义

在 partial blackout 场景下，每个缺失块由三个参数决定[^src-sadi]：

- **缺失特征**：哪些特征（变量）不可用（$\{f_1, \dots, f_k\} \subset \{1,\dots,K\}$）
- **缺失时长**：连续多少个时间步（$\Delta t$）
- **起始时间步**：缺失块开始的位置

一个数据集可包含多个不同的缺失块，每个块的特征集和时长可以不同[^src-sadi]。

## 与其他缺失模式的关系

Partial blackout 是统一的缺失模式框架，将所有常见缺失类型囊括为特例[^src-sadi]：

| 缺失模式 | 在 partial blackout 中的表示 |
|---------|--------------------------|
| 随机缺失 | 多个单特征块，缺失时长 = 1 |
| 插值（单特征连续缺失） | 单特征块，缺失时长 > 1 |
| 完全停电（所有特征缺失） | 全特征块 $k=K$，缺失时长 > 1 |
| 预测任务 | 全特征块 $k=K$，缺失时长 = 预测范围，位于序列末尾 |

## 为什么重要

1. **现实普遍性**：传感器群组故障、区域性通信中断、多指标同步缺失是真实部署中的常态，而非特例[^src-sadi]
2. **暴露模型弱点**：现有方法在随机缺失上表现良好，但在 partial blackout 下性能急剧退化——跨特征依赖建模和长程时间上下文的缺失使问题难度显著增加[^src-sadi]
3. **统一评估框架**：通过控制缺失特征数和时长，可系统评估模型在不同严重程度下的鲁棒性[^src-sadi]

## 实验验证

SADI 在 4 个数据集上系统测试了不同缺失特征数（1-100）和固定缺失时长（10/30 步）的 partial blackout 场景[^src-sadi]。结果表明：

- [[csdi|CSDI]] 在高维 partial blackout 下因分离式特征/时间 Transformer 的信息瓶颈而显著退化
- SADI [[feature-dependency-encoder|FDE]] 显式建模特征间联合时间序列级别依赖，在 partial blackout 下具有决定性优势
- [[mixed-partial-blackout-training|MPB]] 训练策略通过交替暴露于随机缺失和 partial blackout 模式进一步提升鲁棒性

[[stamimputer|STAMImputer]] (arXiv 2025) 针对交通数据的块缺失 (block missing) 设计——多节点在连续时间步同时缺失（如传感器长时间断电），其消融显示这正是 LrSGAT 空间专家最不可替代的场景[^src-stamimputer]。

## 关联页面

- [[sadi]] — SADI，首个针对 partial blackout 设计的扩散插补模型
- [[mixed-partial-blackout-training]] — MPB 训练策略
- [[feature-dependency-encoder]] — FDE，捕获特征间依赖以应对 partial blackout
- [[csdi]] — CSDI，在 partial blackout 下的比较基线
- [[stamimputer]] — STAMImputer，针对交通块缺失设计的时空插补模型
- [[time-series-imputation]] — 时间序列插补方法总览

[^src-sadi]: [[source-sadi]]
[^src-stamimputer]: [[source-stamimputer]]
