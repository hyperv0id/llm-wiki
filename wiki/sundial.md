---
title: "Sundial"
type: entity
tags:
  - time-series
  - foundation-model
  - flow-matching
  - generative-model
  - tsinghua
  - icml-2025
created: 2026-06-08
last_updated: 2026-08-19
source_count: 3
confidence: medium
status: active
---

# Sundial

## 概述

**Sundial** 是清华大学 (BNRist) 提出的首个**原生且灵活**的时间序列基础模型系列，发表于 ICML 2025[^src-sundial]。Sundial 是首个将生成式建模（流匹配）与时间序列基础模型结合的工作，在连续值域中直接预训练，无需离散 tokenization 或指定的参数化先验分布。

## 模型系列

| 规格 | 参数量 | Layers | 维度 | MHA Heads | TimeFlow 网络 |
|------|--------|--------|------|-----------|--------------|
| Sundial-Small | 32M | 6 | (512, 2048) | 8 | (512, 3) |
| **Sundial-Base** | **128M** | **12** | **(768, 3072)** | **12** | **(768, 3)** |
| Sundial-Large | 444M | 24 | (1024, 4096) | 16 | (1024, 6) |

所有规格共享: Patch size=16, Context length ≤ 2880, 预测长度 16 或 720[^src-sundial]。

## 核心创新

### 1. Native & Flexible 设计

Sundial 是 Fig.1 分类中的**原生连续 tokenization + 灵活无先验分布**类别，区别于:
- **Foreign Discrete**: Chronos, LLMTime — 将时序视为"外语"，离散 tokenization 导致 OOV 和粗粒度预测区间
- **Native Parametric**: TimesFM, Timer, Moirai — 连续 tokenization 但使用参数化损失（MSE/分位数）

### 2. TimeFlow Loss

基于 [[flow-matching|Flow Matching]] 的生成式训练目标，让自回归 Transformer 学习每个 patch 的预测分布，能采样生成多样化预测[^src-sundiel]。详见 [[timeflow-loss]]。

### 3. 关键架构提升

- **RoPE**: 提升零样本预测性能 (Fig 9a)
- **Pre-LN**: 更多训练迭代 → 更好性能，Post-LN 反而损害下游结果 (Fig 9b)
- **FlashAttention + KV Cache**: 减少 14.8% 内存占用和 43.6% 推理时间 (Fig 9c-d)
- **Multi-patch prediction**: 减少自回归步数

## 性能亮点

- **TSLib 零样本点预测**: Sundial-Large 8个获胜数第一，超越 Time-MoE (Fig 9, Table 1)
- **GIFT-Eval**: MASE #1, CRPS #2 (23 datasets, 144k 时间序列) 
- **FEV Leaderboard**: 零样本性能超 70% 监督方法，推理速度较 Chronos 快 35×
- **可扩展性**: Sundial-Large 较 Small 训练损失降低 15.38% (Fig 6)
- **测试时校准**: 无需重新训练，增加采样数/采样步数即可提升概率度量

## 与其他基础模型对比

| 维度 | Sundial | Chronos | TimesFM | Moirai | Time-MoE |
|------|---------|---------|---------|--------|----------|
| Tokenization | 连续 Patch | 离散 Point | 连续 Patch | 连续 Patch | 连续 Point |
| 训练目标 | TimeFlow (FM) | Cross-Entropy | MSE | Parametric | Huber |
| 概率预测 | ✓ | ✓ | ✗ | ✓ | ✗ |
| 预训练规模 | 1032B | 94B | 100B | 231B | 300B |
| 推理速度 | 快 (~1s on CPU) | 慢 | 快 | 快 | - |

[^src-sundial] 为简化引用，所有 Sundial 相关论断共享同源

## 相关页面

- [[source-sundial]] — 源文件摘要
- [[timeflow-loss]] — TimeFlow Loss 技术详解
- [[timebench]] — TimeBench 预训练数据集
- [[flow-matching]] — Flow Matching 理论基础
- [[generative-time-series-forecasting]] — 生成式时间序列预测概念
- [[patch-based-tokenization]] — Patch tokenization 技术
- [[chronos]] — Chronos，离散 tokenization 对比
- [[timesfm]] — TimesFM，连续 patch 确定性预测
- [[timer]] — Timer，生成式预训练 Transformer
- [[dits]] — DiTS，MM-DiT + Rectified Flow 用于协变量感知概率预测
- [[cogencast]] — CoGenCast，混合 LLM + FM 编码器-解码器，一步生成预测 (ICML 2026)
- [[cora-tsfm|CoRA]] — Sundial 上的协变量适配框架（CoRA 实验以 Sundial 为主 backbone，MSE 降低 14.2%）
- [[tsfm-covariate-adaptation-comparison]] — TSFM 适配方法全景对比（含 CoRA 基于 Sundial 的详细结果）
- [[hybrid-llm-flow-matching-forecasting]] — 混合 LLM-流匹配预测范式
- [[hybrid-llm-flow-matching-forecasting]] — 混合 LLM-流匹配预测范式
- [[ts-memory]] — TS-Memory：以 Sundial 为冻结 backbone 验证参数记忆蒸馏，Weather 上 MSE 降 2.5%、ETTm1 降 3.6%[^src-ts-memory]
[^src-ts-memory]: [[source-ts-memory-time-series-foundation-models-kdd26]]

[^src-sundial]: [[source-sundial]]
[^src-sundiel]: [[source-sundial]]
