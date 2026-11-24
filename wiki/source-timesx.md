---
title: "Rethinking Multimodal Time-Series Forecasting Evaluation (TimesX)"
type: source-summary
tags:
  - multimodal-time-series
  - benchmark
  - dataset
  - evaluation
  - temporal-leakage
  - time-series-forecasting
created: 2026-11-24
last_updated: 2026-11-24
source_count: 1
confidence: medium
status: active
---

# Rethinking Multimodal Time-Series Forecasting Evaluation

- **来源**：Haoxin Liu、Yichen Zhou、Rajat Sen、B. Aditya Prakash、Abhimanyu Das（原文角标仅标 1/2，未保留机构名；提取核心切片未带会议水印，本仓库 OpenReview CCF-A 调研记录其录用于 ICML 2026）；提出真实时钟无泄露多模态评测基准 **TimesX**[^src-timesx]。
- **核心问题与基准构造**：论文指出，现有合成或缺乏模型知识截断隔离的多模态时序基准不可靠，会让通用大模型甚至代码修订代理排在时序基础模型之上；一旦换为真实数值与真实外部文本并按模型 cutoff 隔离，排序随之反转[^src-timesx]。TimesX 汇集 19 域 × 10 变量共 190 条序列（周频 Google Trends 12 域与日频商品价格/汇率，2018-01 至 2025-10，统一 $L=96, H=12$，含 2024-07-01 起算无泄露主评测集），为每条序列自动构造严格先于预测起点的四类上下文：静态元信息、已知节假日、同域协变量统计摘要，以及经 `Hypothesizer → Verifier → Enricher → Synthesizer` 四步限时检索交叉核查的时间戳新闻事件（50 条人工抽检时间戳精确率 94%）[^src-timesx]。

## 证据

1. **主结论**：真实无泄露评测下模型排序从 `CodeRev > Gemini > TimesFM` 反转为 `TimesFM > Gemini > CodeRev`；越过 2024-06 cutoff 后两类大模型 MASE 恶化约 13%–15%（TSFM 波动不足 2%）；`AvgEns` 稳健优于三类修订代理[^src-timesx]。
2. **归因**：同数值替换 `Time-MMD` 文本使几何均值 MASE 从 0.906 降至 0.840；四类上下文较纯元信息提效约 16%，协变量与事件须成对才有效；`TextRev-ICL` 借历史示例与保守回退超过集成[^src-timesx]。
3. **缺口**：仅覆盖日/周频英文单变量，依赖同域变量领先信号假设，核查约 0.7 美元/变量/三月块[^src-timesx]。

## 相关页面

- [[timesx]] · [[multimodal-time-series-forecasting]] · [[time-mmd]] · [[source-time-imm]] · [[timesfm]] · [[in-context-learning]] · [[doublecast]]

[^src-timesx]: [[source-timesx]]
