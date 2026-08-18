---
title: "TESS (Temporal Evolution Semantic Space)"
type: entity
tags:
  - time-series
  - multimodal
  - llm
  - forecasting
  - semantic-primitives
  - iclr-2026
created: 2026-08-01
last_updated: 2026-08-11
source_count: 1
confidence: medium
status: active
---

# TESS

**TESS**（Temporal Evolution Semantic Space）：Li et al.，arXiv:2603.12664v2，标注 ICLR 2026[^src-tess]。

文本 → 可数值验证的离散时间原语 → 门控 → PatchTST prefix 条件化[^src-tess]。

## 诊断（半合成）

1. $R_t$ 常负：注意力偏冗余 token。
2. 去掉冗余后 Signal-Only 仍显著差于 Numerical-only[^src-tess]。

## 结构

1. 冻结 LLM 分四类 [[temporal-semantic-primitives|原语]]（mean shift / volatility / shape / lag-decay）；margin → 门控 $g_{t,k}$；监督来自 $\psi_k(Y_t)$[^src-tess]。
2. prefix $\Vert$ patch embedding；$L=L_{fcst}+\lambda L_{gate}$[^src-tess]。

理论：4.1 互信息；A.5 误差 $\propto g^2$；A.6 复杂度 $\sqrt{M/n}$[^src-tess]。

## 结果（文内）

Bitcoin vs NewsForecasting：MAE/MSE/RMSE −18.2%/−29.1%/−15.8%。另有 FNSPID / Electricity / Environment；消融见 [[source-tess]][^src-tess]。

## 对照

| | 差在哪 |
|--|--------|
| [[timi\|TiMi]] | 自由文本知识 + MMoE 路由 vs 离散原语 + prefix |
| [[constrained-text-fusion\|CFA]] | 特征层低秩残差 vs 语义层离散瓶颈 |
| [[vot\|VoT]] | 推理链 + 多级对齐 vs 只分类原语 |
| [[time-llm\|Time-LLM]] | TS→LLM vs 文本→TS 模型 |
| [[tats\|TaTS]] | 文本 embedding 拼接 vs 原语+门控 |
| [[patchtst\|PatchTST]] | backbone；多 prefix 与门控 BCE |

## 相关页面

- [[source-tess]] · [[temporal-semantic-primitives]] · [[non-fusion-guidance]] · [[timi]] · [[constrained-text-fusion]] · [[vot]] · [[time-llm]] · [[tats]] · [[patchtst]] · [[multimodal-time-series-forecasting]]

[^src-tess]: [[source-tess]]
