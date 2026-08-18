---
title: "Parametric Memory Distillation"
type: concept
tags:
  - time-series-forecasting
  - knowledge-distillation
  - retrieval-free
  - foundation-model
  - model-adaptation
created: 2026-08-19
last_updated: 2026-08-19
source_count: 1
confidence: medium
status: active
---

# Parametric Memory Distillation

**参数化记忆蒸馏**（Parametric Memory Distillation）是 [[ts-memory|TS-Memory]]（KDD 2026）提出的 TSFM 适配范式——将在线检索蕴含的预测分布知识离线编译为紧凑参数化模块，使冻结 backbone 在推理时获得检索增强的适应能力，无需运行时检索 [^src-ts-memory]。

## 核心思想

在线 kNN 检索不仅提供点估计，还隐含一个非参数条件分布——邻居间的离散度编码了上下文相关的不确定性 [^src-ts-memory]。该分布被视为特权离线监督（privileged supervision），蒸馏到轻量参数模块 PlugMem 中 [^src-ts-memory]。

关键洞察：检索的"知识"（即相似输入暗示的预测分布）可以离线编译为紧凑神经模块，从而将检索成本从推理转移到训练 [^src-ts-memory]。

## 三条适配范式

| 范式 | 机制 | 推理检索 | 推理复杂度 | 灾难性遗忘 |
|------|------|---------|-----------|-----------|
| 参数适配 | 微调/LoRA 更新权重 | 否 | $O(1)$ | 有风险 |
| 非参数检索 | 在线 kNN 搜索 + 融合 | 是 | $O(|\mathcal{D}|)$ | 无 |
| **参数记忆蒸馏** | 离线蒸馏检索分布 → 参数模块 | **否** | $O(1)$ | **无** |

参数记忆蒸馏兼顾前两者的优势：检索增强的鲁棒适应 + 常数时间推理 + 冻结 backbone [^src-ts-memory]。

## 与其他蒸馏范式的关系

- **知识蒸馏（标准）**：从大模型蒸馏到小模型，传递的是模型输出分布
- **特权学习（LUPI）**：训练时使用测试时不可用的特权信息构建辅助目标——TS-Memory 的 kNN 教师即属此类，训练时可访问检索库的未来轨迹，推理时不可 [^src-ts-memory]
- **检索到参数蒸馏**：NLP 领域已有将 RAG 蒸馏为无检索模型的工作；TS-Memory 将此思路迁移到时序预测，蒸馏的是检索引发的分位数校正而非文本生成 [^src-ts-memory]

## 设计要素

1. **泄漏安全教师**：kNN 知识库仅从训练集构建，排除测试窗口泄漏 [^src-ts-memory]
2. **置信门控**：仅当检索教师优于冻结 backbone 时才蒸馏，避免噪声迁移——参见 [[confidence-gated-distillation]] [^src-ts-memory]
3. **分布级蒸馏**：蒸馏分位数目标而非点估计，保留检索带来的不确定性信息 [^src-ts-memory]
4. **推理融合**：参数化记忆与冻结 backbone 的分位数预测线性融合，$\alpha$ 可调 [^src-ts-memory]

## 开放问题

1. 持续记忆更新：域演化下如何增量更新记忆模块而非全量重训 [^src-ts-memory]
2. 更丰富的教师构造：改进可靠性估计和教师目标质量 [^src-ts-memory]
3. 输入自适应融合权重 $\alpha$：当前为全局验证集调优，逐实例自适应留待未来 [^src-ts-memory]

[^src-ts-memory]: [[source-ts-memory-time-series-foundation-models-kdd26]]
