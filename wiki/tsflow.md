---
title: "TSFlow"
type: entity
tags:
  - flow-matching
  - time-series-forecasting
  - gaussian-process
  - probabilistic-model
  - iclr-2025
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# TSFlow

**TSFlow** 是首个将条件流匹配 (Conditional Flow Matching) 应用于概率时间序列预测的模型，发表于 ICLR 2025[^src-tsflow]。由 TU Munich 的 Marcel Kollovieh、Marten Lienen、Leo Schwinn、David Lüdke 和 Stephan Günnemann 提出[^src-tsflow]。

## 核心设计

TSFlow 使用 **高斯过程先验** 替代传统的各向同性高斯先验，将先验分布与时间序列的时序结构对齐[^src-tsflow]。通过最优传输耦合和灵活的条件化策略，TSFlow 同时支持无条件生成和条件预测。

### 关键组件

| 组件 | 描述 |
|------|------|
| **GP 先验** | SE/OU/PE 三种核函数，引入时间相关结构 |
| **最优传输耦合** | mini-batch OT 缩短概率路径、降低训练方差 |
| **条件先验采样** | Langevin 动力学从 $q_0(x_0 \mid y^p)$ 采样 |
| **引导生成** | 通过修改向量场实现无条件模型的条件化 |
| **GP 回归先验** | 条件模型的解析条件先验分布 |

## 架构

TSFlow 使用 DiffWave 风格的残差架构，搭配 S4 层沿时间维度建模时序依赖[^src-tsflow]。3 个残差块，隐藏维度 64，仅约 176k 可训练参数。条件版本额外接收条件向量 $c$（包含观测 $y^p$ 和观测掩码）。时间步通过 64 维正弦位置嵌入编码。通过 Euler ODE 求解器以 32 步采样。

## 性能

在 8 个单变量真实数据集上评估[^src-tsflow]：

- **无条件生成 (W₂)**：GP 先验以 4 NFE 超越 TSDiff（100 NFE）。PE 核在 Solar 数据集上 W₂=4.362 (vs 各向同性 5.193)[^src-tsflow]。
- **下游 LPS**：PE 核在 6/8 数据集上最优[^src-tsflow]。
- **概率预测 (CRPS)**：TSFlow-Cond. 在 6/8 数据集上 SOTA。OU 核在 KDDCup (0.278) 和 Exchange (0.008) 上最佳[^src-tsflow]。
- **vs 扩散模型**：在 7/8 数据集上超越 CSDI、SSSD、TSDiff 和 Biloš et al. (2023)，且 NFE 更少[^src-tsflow]。

## 局限性

- 仅单变量时间序列[^src-tsflow]
- 核函数选择需要验证集调优[^src-tsflow]
- 有条件训练优于无条件+CPS[^src-tsflow]

## 代码

开源：<https://github.com/marcelkollovieh/TSFlow>

## 相关页面

- [[source-tsflow]] — 源文件摘要
- [[flow-matching]] — Flow Matching 理论基础
- [[generative-time-series-forecasting]] — 生成式时间序列预测概念
- [[gaussian-process-prior-flow-matching]] — GP 先验流匹配概念
- [[conditional-prior-sampling]] — 条件先验采样技术
- [[tsdiff]] — TSDiff，TSFlow 的前驱无条件扩散预测模型
- [[optimal-transport]] — 最优传输理论
- [[timegrad]] — TimeGrad，首个时序扩散预测模型
- [[csdi]] — CSDI，条件得分扩散插补/预测
- [[flowts]] — FlowTS，首个 rectified flow TS 生成模型 (arXiv 2025)
- [[rectified-flow-for-time-series]] — Rectified Flow in TS generation
- [[freqflow-ts|FrèqFlow/SpectFlow]] — 频域流匹配 (NeurIPS 2025)
- [[aurora]] — Aurora，原型引导流匹配 (arXiv 2026)

[^src-tsflow]: [[source-tsflow]]
