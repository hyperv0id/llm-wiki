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
last_updated: 2026-08-11
source_count: 3
confidence: medium
status: active
---

# Non-Fusion Guidance

[[timi|TiMi]] 文中的方法标签：LLM 先独立处理外生文本，再以门控/路由引导数值 backbone，不做表示级对齐或 add/concat 融合[^src-timi]。

## 与 Early / Late Fusion

| 标签 | 机制 | TiMi 文中代表 |
|------|------|----------------|
| Early Fusion | TS 进文本/LLM 空间，LLM 作 backbone | [[time-llm\|Time-LLM]], AutoTimes |
| Late Fusion | 分路编码后对齐/映射 | Time-MMD、IMM-TSF、[[vot\|VoT]] |
| Non-Fusion Guidance | 文本支路 → 结构化知识 → MoE 路由时序 experts | TiMi[^src-timi] |

动机（论文）：外生新闻/政策与当前数值往往没有视觉–语言式对齐；直接 fusion 易灌入无关文本[^src-timi]。

## TiMi 怎么做

1. 冻结 LLM 从文本抽出 trends / periodicity / fluctuations 一类结构知识，不直接吐数值预测[^src-timi]。
2. [[mmoe\|MMoE]] 用文本表示门控选 FFN experts（TMoE），另用全局序列门控（SMoE）[^src-timi]。
3. 预测仍由 Transformer 时序主干出[^src-timi]。

## 相近工作

| 工作 | 文本进预测的路径 |
|------|------------------|
| [[tess\|TESS]] | 冻结 LLM → 四类离散 [[temporal-semantic-primitives\|原语]] → 门控 → PatchTST prefix；正文术语是 Temporal Evolution Semantic Space[^src-tess] |
| [[constrained-text-fusion\|CFA]] | 仍特征注入，但门控 / FiLM / 低秩残差限制文本容量[^src-constrained-text-fusion] |
| [[vot\|VoT]] | LLM 推理 + 表示/预测级对齐（Late Fusion 侧）[^src-timi] |

## 相关页面

- [[timi]] · [[mmoe]] · [[vot]] · [[tats]] · [[constrained-text-fusion]] · [[multimodal-time-series-forecasting]] · [[tess]]

[^src-timi]: [[source-timi]]
[^src-constrained-text-fusion]: [[source-constrained-text-fusion]]
[^src-tess]: [[source-tess]]
