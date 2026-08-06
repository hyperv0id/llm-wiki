---
title: "Instance-level Variation"
type: concept
tags:
  - time-series-forecasting
  - instance-level
  - long-tail
  - forecasting-failure
  - distribution-shift
created: 2026-08-06
last_updated: 2026-08-06
source_count: 5
confidence: medium
status: active
---

# Instance-level Variation（实例级变化）

**实例级变化（instance-level variation）** 指时序预测中不同实例（输入-目标对）的误差差异显著、少数实例出现大误差的现象：即使平均指标（如 MSE）整体良好，特定实例的预测仍可能不可靠[^src-pir]。论文（[[pir|PIR]]）将其作为现有模型与整体评估忽视的挑战提出，并据此设计逐实例"识别-修订"流程[^src-pir]。

## 现象证据

论文报告 [[patchtst|PatchTST]] 在 ETTh1 上的逐实例 MSE 曲线（图 1a）：大部分实例误差持续低位，但存在多个尖峰；对应误差分布的直方图与核密度估计呈长尾（图 1b）。论文据此认为实例级变化导致"特定情形下的预测失败"[^src-pir]。

## 论文归纳的来源

论文将实例级变化的来源归纳为三类[^src-pir]：

1. **分布漂移（distribution shifts）**：数据分布随时间或采集环境变化；
2. **缺失值（missing values）**：数据采集过程不完整；
3. **长尾数值模式（long-tail patterns）**：噪声、传感器故障等异常使少数实例呈现罕见数值行为，主流方法难以建模。

论文进一步区分两类失败来源：数据不确定性（如缺失值）与模型不确定性（如长尾模式上的欠拟合）[^src-pir]。

## 与整体评估的关系

论文强调该现象与"整体评估表现良好"并不矛盾：平均指标掩盖了少数实例的失效。因此论文主张在逐实例层面检查预测可靠性，并把"识别并修订失效实例"作为独立研究问题（见 [[post-hoc-forecast-revision]]），而非只优化平均损失[^src-pir]。

## 相关脉络

- **CI/CD 与鲁棒性权衡**：论文引用 channel-independent 策略"以鲁棒性换容量"的讨论（[[channel-independence]]），并报告局部上下文修订对 CI 骨干的收益更大[^src-pir]。
- **长尾与全局模式**：[[gtr|GTR]] 处理"真实周期长于回看窗口、全局周期模式不可见"的问题，与 PIR 的全局修订同属用全局信息覆盖局部建模盲区的路线（周期模式 vs 罕见实例模式）[^src-gtr]。
- **缺失数据家族**：缺失值是论文点名的来源之一。插补方法在数据层面修复缺失（[[csdi|CSDI]] 条件扩散插补[^src-csdi]、[[prdim|PRDIM]] 非随机缺失建模[^src-prdim]、[[nuwats|NuwaTS]] 通用插补基础模型[^src-nuwats]），与 PIR 在预测层面修订失效实例的路线互补。

## 相关页面

- [[pir]] — 提出该现象的框架
- [[post-hoc-forecast-revision]] — 应对该现象的技术路线
- [[error-based-uncertainty-estimation]] — 逐实例误差的估计方法
- [[patchtst]] · [[channel-independence]] · [[gtr]] · [[csdi]] · [[prdim]] · [[nuwats]]

[^src-pir]: [[source-pir]]
[^src-gtr]: [[source-gtr]]
[^src-csdi]: [[source-csdi]]
[^src-prdim]: [[source-prdim]]
[^src-nuwats]: [[source-nuwats]]
