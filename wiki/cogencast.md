---
title: "CoGenCast"
type: entity
tags:
  - time-series
  - llm
  - flow-matching
  - generative-model
  - icml-2026
  - ustc
  - encoder-decoder
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# CoGenCast

**CoGenCast** 是中国科学技术大学（USTC）认知智能全国重点实验室提出的**首个耦合预训练 LLM 与流匹配机制的混合生成式时间序列预测框架**，发表于 ICML 2026[^src-cogencast]。

## 概述

CoGenCast 解决的核心问题是：时间序列预测同时需要语义理解（利用历史模式、领域知识、多模态上下文）和连续随机建模（捕获未来不确定性），而现有的 LLM 方法和流匹配/扩散方法各执一端[^src-cogencast]。通过将预训练 decoder-only LLM（Qwen3 系列）重构为 encoder-decoder 架构，仅修改注意力拓扑，并以 LLM 自回归生成的隐藏状态作为条件驱动流匹配去噪过程，CoGenCast 首次将这两种能力统一到一个框架中[^src-cogencast]。

## 架构

| 组件 | 功能 |
|------|------|
| **LLM Encoder** (Bidirectional SA) | 融合回看窗口 patch + 文本上下文特征（领域、任务指令、统计信息），产生双向上下文编码 |
| **LLM Decoder** (Causal SA + Cross-Attention) | 自回归生成未来 patch 表示，以 encoder 输出为条件，保持严格因果性 |
| **Denoising Decoder** (Flow Matching) | 以 LLM decoder 隐藏状态为条件，预测区间条件化平均速度场，一步生成 |

## 关键性能

- **10 数据集 SOTA**：Energy, ETT (4 subsets), Environment, Exchange, Health, Wind, Solar 上全面领先[^src-cogencast]
- **MSE ↓11%** vs LLM 类基线（LLM4TS, Time-LLM）；**MSE ↓7%** vs Transformer 类基线（PatchTST, Autoformer）[^src-cogencast]
- **跨域训练增益**：多领域联合训练进一步提升泛化性能[^src-cogencast]
- **一步生成**：单次函数求值完成推理，低延迟[^src-cogencast]
- **不确定性量化**：50%/80% 预测区间紧密包裹真实值[^src-cogencast]
- **Backbone 可扩展**：Qwen3-0.6B 默认（效率最优），Qwen3-4B 最高精度[^src-cogencast]

## 核心设计决策

| 维度 | 选择 |
|------|------|
| LLM Backbone | Qwen3-0.6B（效率最优） |
| Noise Scheduler | Linear（配合直线轨迹最优） |
| Patch Size | 4-6（数据集依赖） |
| 采样步数 | 1 步（NFE=1） |
| 上下文特征 | 领域 + 任务指令 + 统计信息 |
| 代码 | [github.com/liuyaguo/_CoGenCast](https://github.com/liuyaguo/_CoGenCast) |

## 消融关键发现

- **Encoder-only vs Decoder-only vs Full**：完整 encoder-decoder 最优；decoder-only 严重退化，证明双向编码器对语义理解至关重要[^src-cogencast]
- **w/o AR vs w/o Flow vs w/o Both**：移除任一组分均导致显著性能下降，两者协同是核心驱动力[^src-cogencast]
- **w/o Text**：移除全部文本上下文导致全面退化；统计信息（均值/方差）是最关键的文本子组件[^src-cogencast]
- **Linear vs Cosine Scheduler**：Linear 显著优于 Cosine，因为直线轨迹与平均速度建模天然对齐[^src-cogencast]
- **NFE Ablation**：一步生成已达最优或接近最优，2-3 步仅边际增益[^src-cogencast]

## 关系

- 改进于：[[time-llm]] — Time-LLM 的模型重编程思路，但 CoGenCast 将冻结 LLM 改为微调 encoder-decoder 架构
- 并行于：[[sundial]] — Sundial 使用纯 Transformer + TimeFlow Loss，CoGenCast 混合 LLM + FM
- 并行于：[[flowts]] — FlowTS 使用纯 rectified flow，CoGenCast 增加 LLM 语义条件
- 概念基础：[[hybrid-llm-flow-matching-forecasting]] — 混合 LLM-FM 预测范式
- 技术基础：[[flow-matching]], [[one-step-flow-generation]], [[average-velocity-modeling]]
- 对比：[[generative-time-series-forecasting]] — 生成式时间序列预测全景

[^src-cogencast]: [[source-cogencast]]