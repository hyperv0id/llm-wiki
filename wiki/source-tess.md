---
title: "From Text to Forecasts: Bridging Modality Gap with Temporal Evolution Semantic Space"
type: source-summary
tags:
  - time-series
  - multimodal
  - llm
  - forecasting
  - iclr-2026
created: 2026-08-01
last_updated: 2026-08-11
source_count: 0
confidence: low
status: active
---

# From Text to Forecasts: Bridging Modality Gap with Temporal Evolution Semantic Space

**Authors**: Lehui Li et al.（山大 / 波士顿大学 / 北航 / 南科大）。**标注**: ICLR 2026。**arXiv**: 2603.12664v2。

## 问题与诊断

文本对事件影响偏定性、弱时间锚定。半合成（FNSPID + GPT-5.2 对齐文本，token 标注）[^src-tess]：

1. 焦点比 $R_t=\log(\bar\alpha_{sig}/\bar\alpha_{red})$ 常 $<0$（注意力偏冗余 token）。
2. Signal-Only > Full，但仍远差 Numerical-only。

## 方法 TESS

1. 冻结 LLM：文本 → 四类 [[temporal-semantic-primitives|原语]]（mean shift / volatility / shape / lag-decay）；margin 门控；门控标签由未来窗口数值可验证性自动生成[^src-tess]。
2. 门控后 embedding 作 prefix，拼 PatchTST patches；$L=L_{fcst}+\lambda L_{gate}$，LLM 冻结[^src-tess]。

定理 4.1 / A.5 / A.6：充分性下互信息、误差按 $g^2$ 衰减、复杂度 $\sqrt{M/n}$（见原文前提）[^src-tess]。

## 结果（四数据集）

| 设定 | 相对最强基线（文内） |
|------|----------------------|
| Bitcoin vs NewsForecasting | MAE/MSE/RMSE −18.2% / −29.1% / −15.8% |
| FNSPID vs TimesNet | −3.3% / −20.0% / −9.9% |
| Electricity | 全指标最优 |
| Environment | 次优 |

消融：去 TESS 组件 MSE +46.2%/+29.4%/+22.8%；去 gating +3.7%/+2.6%/+7.5%；去 mean shift +33%。正文消融% 与 Table 2 不完全一致时以表为准[^src-tess]。

未披露冻结 LLM 型号；代码“接收后发布”。附录 Table 3 FNSPID 粒度栏印 “5 days”，与 daily 表题矛盾（原文如此）[^src-tess]。

[^src-tess]: [[source-tess]]
