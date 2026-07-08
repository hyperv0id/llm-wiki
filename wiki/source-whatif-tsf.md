---
title: "What If TSF: A Benchmark for Reframing Forecasting as Scenario-Guided Multimodal Forecasting"
type: source-summary
tags:
  - benchmark
  - counterfactual
  - multimodal
  - spatiotemporal
  - 2026
created: 2026-07-07
last_updated: 2026-07-07
source_count: 1
confidence: medium
status: active
---

# What If TSF (WIT) Benchmark

**作者：** Jinkwan Jang, Hyunbin Jin, Hyungjin Park, Kyubyung Chae, Taesup Kim（首尔大学）
**年份：** 2026 | **arXiv：** 2601.08509

## 核心贡献

What If TSF（WIT）是一个创新的场景引导多模态时序预测基准。传统时序预测模型仅依赖历史模式外推，假设未来与过去相似；而人类专家会构建"what-if"场景来推演不同结果——同一历史数据在不同场景下可能产生截然不同的预测。WIT 受此启发，通过提供专家构建的可能场景和反事实场景，评估模型能否根据文本上下文调整预测方向。现有基准（TS-Insights、ChatTS、MoTime、MTBench、TSQA、Time-MMD、CiK 等）普遍存在信息重复、格式不一致、记忆化风险和时间错位等问题，无法有效评估多模态预测的真实能力。[^src-whatif-tsf]

## 基准设计

WIT 包含 5,352 个样本，覆盖政治（总统批准率等）、社会（移民意见等）、能源（原油库存等）和经济（汇率等）四个领域。每个样本包含四类输入：（1）数值时序历史 x₁₋ₜ；（2）静态上下文 S（领域和变量描述）；（3）历史上下文 H（过去事件的摘要，过去时态）；（4）未来场景 F——可能场景 Fpl 或反事实场景 Fcf。未来场景采用条件化语言（"如果""可能""预计"等），避免直接透露答案，仅描述可能影响目标变量的外部因素变化。[^src-whatif-tsf]

基准定义三项核心任务：短时预测（ST）评估短视界方向准确率和 MSE 数值精度；长时预测（LT）仅评估 3 类方向准确率（上升/不变/下降），因长视界数值预测不确定性过高；反事实预测（CF）在历史数据不变下替换为对立场景，采用最小编辑设计检验模型能否翻转方向判断，仅能评估方向准确率。[^src-whatif-tsf]

## 实验评估

评估涵盖多种 LLM（Mistral-7B、Qwen2.5-7B、Mixtral-8x22B、Gemma-3-27B、Qwen3-32B、Llama-3-70B、GPT-4o），时序基础模型 TSFM（Chronos-Bolt-Base、Moirai-1.1-R-Large、TimesFM-2.5-200M）和统计方法（ARIMA、ETS、Exponential Smoothing）。核心发现：（1）加入未来场景 Fpl 后方向准确率大幅提升，GPT-4o 从仅 S 的 50.4% 提升至 78.4%（+28 个百分点），而历史上下文 H 的贡献极为有限（仅 0-3%）；（2）ST 和 CF 任务表现高度一致，说明模型能根据反事实场景有效区分对立结果方向；（3）TSFM 和统计方法在 MSE 上可比但方向准确率远低于 LLM，因无法利用外部文本信息；（4）知识截止分析验证了去标识化的有效性。[^src-whatif-tsf]

## 局限

WIT 的未来场景由专家手工构建，对强推理 LLM 可扩展为自主生成场景，但需解决幻觉控制和评估分离。部分 LLM（Llama-3）在数值预测中违反基本值域约束，暗示 LLM 对时序语义理解仍需深入。少样本提示和多步推理是未来方向。[^src-whatif-tsf]

## 交叉链接

- [[source-terra]] — Terra 大规模多模态地球时空数据集
- [[multimodal-time-series-forecasting]] — 多模态时序预测综述

[^src-whatif-tsf]: [[source-whatif-tsf]]
