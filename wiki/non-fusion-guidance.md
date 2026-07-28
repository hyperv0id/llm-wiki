---
title: "Non-Fusion Guidance"
type: concept
tags:
  - multimodal
  - time-series
  - llm
  - forecasting
  - mixture-of-experts
created: 2026-07-25
last_updated: 2026-07-28
source_count: 2
confidence: medium
status: active
---

# Non-Fusion Guidance

**Non-Fusion Guidance** 是 [[timi|TiMi]] (ICML 2026) 提出的第三种多模态时间序列预测范式，区别于传统的 Early Fusion 和 Late Fusion[^src-timi]。

## 三种范式对比

| 范式 | 机制 | 代表方法 | 核心问题 |
|------|------|---------|----------|
| **Early Fusion** | TS → 文本空间，LLM 作 backbone | [[time-llm|Time-LLM]], AutoTimes | LLM 不适合直接做数值预测 |
| **Late Fusion** | 分别编码，表示级对齐+映射 | Time-MMD, IMM-TSF | 文本-数值缺乏语义对应 |
| **Non-Fusion Guidance** | LLM 独立推理 → MoE 路由引导 TS backbone | **TiMi** | 依赖 LLM 推理质量 |

## 核心洞察

Early/Late Fusion 都试图在**表示层**对齐或融合文本与时间序列，但这忽略了关键事实：外生文本（新闻、政策公告）与数值序列**缺乏直接的语义对应**——文本描述的是未来期望而非当前状态[^src-timi]。元数据文本（如数据集描述）虽然与序列语义对齐，但只反映序列内部属性，无法引入外部因果因素[^src-timi]。

## 原理

Non-Fusion Guidance 将文本和时序建模**完全解耦**[^src-timi]：

1. **独立推理**：冻结 LLM 独立于时序 backbone 处理文本，生成未来趋势的结构化因果知识（trends, periodicity, fluctuations），而非数值预测[^src-timi]。
2. **知识引导**：通过 [[mmoe|MMoE]] 的门控机制，将文本因果知识作为路由信号选择时序 experts，而非直接融合特征[^src-timi]。
3. **时序主导**：时间序列 backbone 保持完整的功能，文本仅提供辅助性预测引导[^src-timi]。

这一设计无需模态对齐，天然支持**不规则多模态数据**（文本与数值采样频率不同、时间戳不对齐），因为两个模态的处理完全独立[^src-timi]。

## 与相关方法的差异

- **vs VoT**：[[vot|VoT]] 也使用 LLM 推理，但最终通过多级对齐（表示级+预测级）融合文本与数值特征，属于 Late Fusion[^src-timi]。
- **vs TaTS**：[[tats|TaTS]] 将文本编码后作为辅助变量拼接至数值序列，虽然即插即用但仍是特征层面的简单融合，未利用因果推理[^src-timi]。
- **vs Constrained Text Fusion / CFA**：[[constrained-text-fusion|CFA]] 与本文同认「naive add/concat 常低于 unimodal」，但选择 **受控特征融合**（低秩残差 / 门控 / FiLM / 正交）而非完全解耦；TiMi 走 **不融合、只路由**，CFA 走 **过滤后仍残差注入 TS 表示**——二者是互补的反-naive 路线[^src-constrained-text-fusion][^src-timi]。

## 相关页面

- [[timi]] · [[mmoe]] · [[vot]] · [[tats]] · [[constrained-text-fusion]] · [[source-constrained-text-fusion]] · [[multimodal-time-series-forecasting]]


[^src-timi]: [[source-timi]]
[^src-constrained-text-fusion]: [[source-constrained-text-fusion]]
