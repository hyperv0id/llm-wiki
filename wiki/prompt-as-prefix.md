---
title: "Prompt-as-Prefix"
type: technique
tags:
  - time-series
  - llm
  - prompting
  - reprogramming
created: 2026-06-04
last_updated: 2026-09-13
source_count: 2
confidence: medium
status: active
---

# Prompt-as-Prefix (PaP)

Prompt-as-Prefix (PaP) 是 Time-LLM 中用于增强冻结 LLM 对时间序列推理能力的提示技术。与将时间序列直接翻译为自然语言的 Patch-as-Prefix 不同，PaP 将自然语言提示前置在 reprogrammed patches 之前，引导 LLM 的变换而不要求 LLM 直接生成数值 [^src-time-llm]。

## 动机

直接让 LLM 预测数值（Patch-as-Prefix）有两个关键缺陷 [^src-time-llm]：
1. LLM 对高精度数值的敏感度不足，难以在长预测范围内准确输出
2. 不同 LLM 的 tokenization 差异导致数值表示不统一（如 0.61 可能被表示为 ['0','.','6','1'] 或 ['0','.','61']），需要复杂的后处理

PaP 通过让 LLM 输出 hidden representations 而非直接数值来避免这些问题 [^src-time-llm]。

## 三部分结构

### 1. Dataset Context
提供输入时序的背景信息，帮助 LLM 理解数据领域特征 [^src-time-llm]：
> "The Electricity Transformer Temperature (ETT) indicates the electric power long-term deployment. Each data point consists of the target oil temperature and 6 power load features…"

### 2. Task Instruction
明确指示 LLM 的处理目标 [^src-time-llm]：
> "Predict the next H steps given the previous T steps information attached"

### 3. Input Statistics
最关键的组件（移除后 MSE ↑10.2%）[^src-time-llm]：
> "The input has a minimum of `<min>`, a maximum of `<max>`, and a median of `<median>`. The overall trend is `<upward/downward>`. The top five lags are `<lag_values>`."

## 实验证据

- 移除 PaP → 全量预测 8% MSE 退化，few-shot 超过 19% [^src-time-llm]
- 三个子组件重要性排序：Input Statistics > Dataset Context (9.6%) > Task Instruction (7.7%) [^src-time-llm]
- PaP + Patch Reprogramming 联合作用显著优于各自单独 [^src-time-llm]

## 与推理轨迹前置的区别

[[trace-as-state|Trace as State]] 也把文本放在待处理内容之前，但文本来源不是预先组织的数据集说明、任务指令与输入统计，而是模型读过**同一道题**后生成的推理轨迹；下一遍仍读完整原文，并与使用同一轨迹的后置条件比较（§3.2–3.3）。[^src-trace-as-state]

[INFERENCE] PaP 与 Trace as State 可按“前置信息从哪里来”作设计对照：前者组织已有的任务/统计信息，后者回收前一遍新发现的任务信息。二者没有在同一时序预测设置下直接比较；不能仅因都使用 prefix 就推断性能等价或可直接替换。[^src-time-llm][^src-trace-as-state]

## Connections

- 属于：[[time-llm]] — Time-LLM 框架的核心组件
- 对比：[[patch-reprogramming]] — Time-LLM 的另一核心组件
- 关系：[[timecap]] — 同样利用 LLM 的提示能力处理时序，但目的不同（事件预测 vs 数值预测）
- [[trace-as-state]] — 同题 reasoning text 作为 prefix 的跨遍方法

[^src-time-llm]: [[source-time-llm]]
[^src-trace-as-state]: [[source-trace-as-state]]
