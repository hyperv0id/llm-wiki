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
last_updated: 2026-11-27
source_count: 1
confidence: medium
status: active
---

# Rethinking Multimodal Time-Series Forecasting Evaluation (TimesX)

要让文本真正帮上时序预测的忙，先得有一份不会作弊的考卷。论文指出，现有多模态时序基准要么依赖小规模合成数据，要么无法隔离「模型训练时是否早已见过这段数据」，在这样的考卷上，通用大模型甚至代码修订代理会排到专用时序基础模型之上，排名被系统性虚高。TimesX 的核心巧思是全盘改用真实观测数值，并给每条序列配上由自动管线生成、经多步限时联网检索交叉核查、发布时间严格早于预测起点的文本上下文；同时所有评测样本的预测起点都落在参评模型的知识截断之后，基准还能随新截断持续刷新，从构造上杜绝泄露。换上这份无泄露考卷后，结论随之反转：时序基础模型重回领先，大模型越过知识截断后误差明显恶化，而简单集成稳健优于无训练修订代理[^src-timesx]。

## 核心做法

- **无泄露真实时钟评测集**：汇集 19 域 × 10 变量共 190 条真实单变量序列（周频 Google Trends 12 域与日频商品价格、汇率，2018-01 至 2025-10，统一 $L=96$、$H=12$）；每个评测窗口的预测起点严格晚于全部参评模型的知识截断，2024-07-01 起算的子集构成无泄露主评测集，另划 2018–2023 年段供后续微调训练使用，基准可随新截断持续刷新[^src-timesx]。
- **四类前置上下文**：为每条序列自动构造静态元信息、已知节假日日历、同域协变量统计摘要（同域其余序列的均值、极值与近期趋势压缩为自然语言）、带时间戳的新闻事件四类文本；事件只取发布时间严格早于预测起点、最近的 $k=10$ 条，避免未来信息泄入[^src-timesx]。
- **多步限时检索核查管线**：事件上下文由四个标准步骤串联生成——先由模型扫描序列、围绕异常波动提出候选事件假设（Hypothesizer），再按时间窗约束联网检索并逐条核实事实与日期（Verifier），对存疑信息做跨源裁决与补全（Enricher），最后将通过核查的材料合成为时间戳标注文本（Synthesizer）；全程强制程序化时间范围约束，50 条人工抽检的事件时间戳精确率 94%[^src-timesx]。
- **上下文示例驱动的保守修订（TextRev-ICL）**：修订代理不让大模型凭空改写预测，而是检索一条时间上邻近的历史修订案例作为示范，并显式提供「证据不足则维持原预测」的回退选项；保守回退是修订代理超过简单集成的关键[^src-timesx]。

## 证据

1. **定位与主结论**：无泄露评测下模型排序反转，时序基础模型重回领先；两个大模型越过 2024-06 知识截断后 MASE 恶化约 12%–15%，时序基础模型同期波动不足 2%；免训练简单集成稳健优于三类修订代理[^src-timesx]。
2. **真实涨点归因**：固定数值、仅把 Time-MMD 旧文本换成 TimesX 核查文本，几何均值 MASE 相对下降约 7%；四类上下文较仅给元信息提效约 16%，但协变量摘要与新闻事件须成对提供；历史示例加保守回退（TextRev-ICL）是修订代理超过集成的关键[^src-timesx]。
3. **证据缺口**：仅覆盖日/周频英文单变量；隐含「同域协变量存在领先信号」假设，并非所有域成立；检索核查成本约 0.7 美元/变量/三月块，制约刷新与复用[^src-timesx]。

## 相关页面

- [[timesx|TimesX]] — 实体页：本基准的汇总页与跨工作对照枢纽。
- [[multimodal-time-series-forecasting|多模态时序预测]] — 跨槽位正交：本基准构成该方向的多模态评测基础设施。
- [[time-mmd|Time-MMD]] — 复现对齐：同数值序列仅替换文本的受控实验，以其旧基准文本为对照面。
- [[timesfm|TimesFM-2.5]] — 跨槽位正交：无泄露评测下重回领先的时序基础模型代表，简单集成中承担数值外推。
- [[in-context-learning|上下文学习]] — 跨槽位正交：TextRev-ICL 将上下文学习置于预测后修订槽位，以历史示例加保守回退抑噪。

出版信息：ICML 2026（PMLR 306）；Haoxin Liu、B. Aditya Prakash（Georgia Tech），Yichen Zhou、Rajat Sen、Abhimanyu Das（Google Research）。

[^src-timesx]: [[source-timesx]]
