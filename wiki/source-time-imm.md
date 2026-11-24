---
title: "Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series"
type: source-summary
tags:
  - multimodal-time-series
  - irregular-time-series
  - benchmark
  - dataset
  - time-series-forecasting
  - neurips-2025
created: 2026-11-24
last_updated: 2026-11-24
source_count: 2
confidence: medium
status: active
---

# Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series

**来源**：Ching Chang 等（UCLA、NYCU）；NeurIPS 2025 Datasets and Benchmarks Track（D&B）；提出数据集库 **Time-IMM** 与基准评测框架 **IMM-TSF**[^src-time-imm]。

真实世界的多变量时间序列常面临采样不均、各通道时间戳异步以及离散文本日志伴随出现的问题，而既有预测基准多假设规则时间网格或已完成模态对齐。针对这一缺口，Time-IMM 依据触发、约束、故障三类现实成因构建了九个亚类的异步多模态数据集，并在保留各通道与文本独立时间戳的前提下，通过学习软融合来系统评估非均匀多模态时序的预测规律[^src-time-imm]。

## 数据集设计与基准定位

Time-IMM 按不规则性的生成成因分为**触发**、**约束**与**故障**三类共九个亚类数据集，涵盖 GDELT、RepoHealth、MIMIC、FNSPID、ClusterTrace、StudentLife、ILINet、CESNET 与 EPA-Air[^src-time-imm]。为压缩文本长度并剔除冗余表述，原始文本日志统一由大语言模型筛选并提炼为五句摘要[^src-time-imm]。

在基准演进谱系上，Time-IMM 与 [[time-mmd|Time-MMD]] 构成清晰的**瓶颈接力 $\Delta$**：[[time-mmd|Time-MMD]] 构建了多领域数值–文本基准并控制了未来信息泄漏，但仍假设规则的日、周、月网格与强时间对齐；Time-IMM 则进一步放宽到非均匀采样与通道异步场景，免除了模态与通道间的强对齐前提[^src-time-imm]。同时，与同样补齐 [[time-mmd|Time-MMD]] 的 [[timesx|TimesX]] 形成**同槽位基准对照**：Time-IMM 拓展了非均匀异步日志维度，而 TimesX 聚焦规则日/周频序列的四类真实时钟上下文与严格模型 cutoff 时间隔离[^src-time-imm][^src-timesx]。

## 评测框架与算子拆解（IMM-TSF）

论文配套的 IMM-TSF 是由既有标准算子拼装而成的异步多模态评测框架，未引入新算子或新理论：对于需要规则输入的常规数值模型，先以全局时间网格与观测掩码进行预对齐；针对不规则时间戳的原生时间融合，框架对比了**高斯近因平均**与**带时间编码的交叉注意力**两条路径；在数值与文本的模态融合槽位，则对比了**门控循环残差相加**与**固定权重交叉注意力相加**两种方案[^src-time-imm]。这种对受控门控融合的实证偏好也与 [[constrained-text-fusion|受控文本融合]] 的发现相呼应[^src-time-imm]。

## 证据

1. **定位与主结论**：加入文本平均降误差约 7%，强相关场景近 38%；纯数值骨干以分块图网络最强，收益取决于文本相关性[^src-time-imm]。
2. **真实涨点归因**：门控循环残差相加优于固定权重交叉注意力相加（抑噪）；两种时间融合与不同规模冻结编码器间无明显差异[^src-time-imm]。
3. **证据缺口与负结果**：仅测已知查询时刻预测，缺复合不规则与非预测任务，亦未消融未经摘要的原始文本及微调文本编码器[^src-time-imm]。

## 相关页面

- [[multimodal-time-series-forecasting]] — 多模态时间序列预测概念与基准比较
- [[time-mmd]] — 前序规则网格数值–文本基准（瓶颈接力 $\Delta$）
- [[multimodal-semantic-understanding]] — 模态对齐、门控融合与文本冗余过滤分析
- [[constrained-text-fusion]] — 文本受控注入与噪声抑制的跨源实证对齐
- [[timesx]] — 规则日/周频四类上下文与模型 cutoff 隔离基准（同槽位并行对照）

[^src-time-imm]: [[source-time-imm]]
[^src-timesx]: [[source-timesx]]
