---
title: "Prototype-Guided Flow Matching"
type: technique
tags:
  - flow-matching
  - generative-model
  - time-series-forecasting
  - probabilistic-forecasting
  - arxiv-2026
created: 2026-05-03
last_updated: 2026-05-03
source_count: 2
confidence: high
status: active
---

# Prototype-Guided Flow Matching

**Prototype-Guided Flow Matching** 是 [[aurora|Aurora]] 中提出的生成式概率预测技术，用于在解码阶段生成未来时间序列 token[^src-aurora]。

## 动机

标准 [[flow-matching|Flow Matching]] 通过从噪声到数据的向量场回归实现生成建模，但在时间序列预测中，未来值的生成需要以历史观测和多模态上下文为条件。Aurora 通过引入"原型"（prototypes）来引导流匹配过程，使生成更符合领域特定的未来趋势[^src-aurora]。

## 机制

在解码阶段，Aurora 的多模态表示用于生成两个关键组件[^src-aurora]：

1. **条件（Conditions）**：基于多模态领域知识生成的条件向量，约束流匹配的生成方向
2. **原型（Prototypes）**：未来 token 的典型模式表示，作为流匹配的目标锚点

这些条件和原型共同引导 Flow Matching 过程，使生成的概率预测既符合历史时间序列模式，又融入多模态领域知识。

## 与标准 Flow Matching 的区别

| 维度 | 标准 Flow Matching | Prototype-Guided Flow Matching |
|------|-------------------|-------------------------------|
| 条件 | 无条件或简单条件 | 多模态领域知识条件 |
| 目标 | 从噪声到数据 | 从噪声到原型引导的未来分布 |
| 应用 | 图像/音频生成 | 时间序列概率预测 |
| 引导 | 无 | 原型 + 条件双重引导 |

## 与 SimDiff 扩散方法的对比

[[simdiff|SimDiff]] 使用 DDPM 扩散模型进行点预测，通过 Median-of-Means 将概率样本聚合为点估计[^src-simdiff]。Aurora 的 Prototype-Guided Flow Matching 则直接进行概率预测，保留完整的预测分布信息，且使用更高效的 Flow Matching（OT 路径直线轨迹）替代扩散路径[^src-aurora]。

## 相关页面

- [[aurora]] — Aurora 模型
- [[modality-guided-self-attention]] — 编码阶段的模态引导注意力
- [[flow-matching]] — Flow Matching 理论基础
- [[generative-time-series-forecasting]] — 生成式时间序列预测概念
- [[simdiff]] — 扩散式生成预测对比

[^src-aurora]: [[source-aurora]]
[^src-simdiff]: [[source-simdiff]]
