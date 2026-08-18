---
title: "TiMi"
type: entity
tags:
  - time-series
  - multimodal
  - mixture-of-experts
  - llm
  - forecasting
  - icml-2026
created: 2026-07-25
last_updated: 2026-08-11
source_count: 3
confidence: medium
status: active
---

# TiMi

**TiMi**（Time Series Transformers with Mixture of Experts）：清华，arXiv:2602.21693，标注 ICML 2026[^src-timi]。

冻结 LLM 抽外生文本结构知识 → **[[mmoe|MMoE]]** 换 FFN 路由 → Transformer 时序主干预测[^src-timi]。论文方法标签：[[non-fusion-guidance|Non-Fusion Guidance]][^src-timi]。

## 对照

| | 机制 |
|--|------|
| Early Fusion（Time-LLM 等） | TS 进 LLM 空间 |
| Late Fusion（Time-MMD、[[vot\|VoT]]） | 分路编码后对齐/映射 |
| TiMi | 文本知识 → MoE 路由，无表示融合[^src-timi] |

- vs [[vot|VoT]]：都用 LLM 读外生文本；VoT 做多级对齐，TiMi 做路由[^src-timi]。
- vs [[constrained-text-fusion|CFA]]：CFA 低秩/门控特征注入；TiMi 不做特征融合[^src-constrained-text-fusion]。
- vs [[tess|TESS]]：TESS 是 4 类离散原语 + 门控 + PatchTST prefix；术语是 Temporal Evolution Semantic Space[^src-tess]。

## 架构

1. 冻结 Qwen2.5-7B-Instruct 处理文本 → 池化 token[^src-timi]
2. Patch embedding（PatchTST 风格）[^src-timi]
3. MMoE：TMoE（文本门控 Top-K）+ SMoE（全局序列门控 Top-K）[^src-timi]

$$\text{MMoE}(h,\bar H)=\sum_{i\in\tau_x}s_{i,x}\mathrm{FFN}_i(h)+\sum_{i\in\tau_s}s_{i,s}\mathrm{FFN}_i(h)$$

## 结果（文内 16 基准）

| 项 | 数字 |
|----|------|
| PatchTST / TimeXer / Autoformer +MMoE | 平均 MSE −18.2% / −12.5% / −12.4% |
| Time-IMM vs PatchTST | 平均 MSE −29.57% |
| Time-MMD vs PatchTST | 平均 MSE −11.26%[^src-timi] |

SMoE 路由与 MK 趋势方向相关[^src-timi]。

## 相关页面

- [[source-timi]] · [[non-fusion-guidance]] · [[mmoe]] · [[vot]] · [[time-mmd]] · [[constrained-text-fusion]] · [[multimodal-time-series-forecasting]] · [[tess]]

[^src-timi]: [[source-timi]]
[^src-constrained-text-fusion]: [[source-constrained-text-fusion]]
[^src-tess]: [[source-tess]]
