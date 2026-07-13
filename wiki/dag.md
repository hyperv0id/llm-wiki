---
title: "DAG"
type: entity
tags:
  - exogenous-variables
  - correlation
  - attention
  - time-series
  - ijcai-2026
created: 2026-07-12
last_updated: 2026-07-13
source_count: 1
confidence: high
status: active
---

# DAG

**DAG**（**D**ual correl**A**tion **G**etwork；全称 *A Dual Correlation Network for Time Series Forecasting with Exogenous Variables*）是华东师范大学 Decision Intelligence 组（与 [[source-timexer|TimeXer]] / DUET / TFB 同组）提出的确定性外生预测框架，发表于 IJCAI 2026。[^src-dag]

代码：<https://github.com/decisionintelligence/DAG>

## 要解决什么

TSF-X 任务中，现有方法两大短板：[^src-dag]

1. **忽略未来外生**：TimeXer、CrossLinear 只用历史外生，浪费已知未来协变量信息。
2. **无相关性建模**：TiDE、TFT 简单拼接未来外生，无内外生相关性约束，易陷伪相关。

## 核心洞察

外生预测中存在**双相关结构**（Figure 2）：[^src-dag]

- **时间维**：历史外生 → 未来外生（Granger 因果），结构相似于历史内生 → 未来内生。
- **通道维**：历史外生 ↔ 历史内生（Pearson 相关），模式可迁移到未来外生 ↔ 未来内生。

## 架构

四网络两模块，每个模块含"发现 + 注入"：[^src-dag]

| 模块 | 发现 | 注入 |
|------|------|------|
| 时间相关 | F_θ1: 历史外生 → 预测未来外生，提取 Wq',Wk' | G_θ2: 历史内生 → 预测未来内生，注入 Wq',Wk' |
| 通道相关 | F_θ3: 历史外生 → 预测历史内生，提取 Wq',Wk' | G_θ4: 未来外生 → 预测未来内生，注入 Wq',Wk' |

注入机制见 [[dual-correlation-injection]]。[^src-dag]

## 结果速览

- 12 个 TSF-X 数据集：MSE 10/12 第一，MAE 11/12 第一。[^src-dag]
- 无未来外生时用 F_θ1 预测 Ŷ_exo 替代，仍优于纯历史外生方法。[^src-dag]
- 消融确认双相关协同最优。[^src-dag]

## 与邻近工作的位置

- **相对 TimeXer**：DAG 显式建模并利用未来外生，而非仅 patch 融合历史外生。[^src-dag]
- **相对 TiDE/TFT**：DAG 不做简单拼接，而是发现—注入相关性结构，抑制伪相关。[^src-dag]
- **相对 [[kite|KITE]]**：DAG 是确定性预测 + 学习相关性注入；KITE 是概率预测 + 统计先验注入（[[knowledge-guided-conditioning|KGC]]）+ [[classifier-free-guidance|CFG]] 剂量控制 + [[history-conditional-manifold|HCM]] 源几何。两者在"用相关性指导外生建模"同向，但相关性来源与预测范式不同。[^src-dag]
- **相对 [[gcgnet|GCGNet]]**：同为确定性外生预测；DAG 做双相关发现—注入到注意力，GCGNet 用 VAE 粗生成 + 图结构对齐 + GCN 精炼做联合时间–通道图一致建模。[^src-dag]

## 相关页面

- [[source-dag]] — 源摘要
- [[dual-correlation-injection]] — 相关性发现—注入技术
- [[kite]] / [[source-kite]] — 概率外生预测对照
- [[gcgnet]] / [[source-gcgnet]] — 图一致生成外生预测对照（ICLR 2026）
- [[source-timexer]] / [[source-exotst]] / [[source-exost]] / [[source-select-then-balance]]
- [[cross-attention-conditioning]]
- [[covariate-fusion-module]]

[^src-dag]: [[source-dag]]
