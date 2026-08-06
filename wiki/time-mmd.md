---
title: "Time-MMD"
type: entity
tags:
  - multimodal-time-series
  - dataset
  - benchmark
  - text-numeric-alignment
  - neurips-2024
created: 2026-07-28
last_updated: 2026-08-06
source_count: 4
confidence: high
status: active
---

# Time-MMD

**Time-MMD**（Multi-Domain Multimodal Dataset for Time Series Analysis）是 NeurIPS 2024 D&B 发布的**多领域数值–文本时序数据集**（arXiv:2406.08627v4），配套预测库 **MM-TSFlib**。定位：填补“窄领域 / 粗对齐 / 文本污染”的多模态 TSA 数据空白，成为后续文本协变量与多模态 TSF 的标准基准之一[^src-time-mmd]。

## 设计目标与缺口

相对金融新闻–股价类数据，Time-MMD 针对三点：

1. **多领域**：9 个主域，数值模式与文本稀疏度各异[^src-time-mmd]。
2. **细粒度模态对齐**：报告/关键词围绕**选定目标变量**筛选与过滤，而非同域粗匹配[^src-time-mmd]。
3. **污染控制**：LLM 将文本拆为 **fact vs prediction**；cutoff 推至 **2024-05**，降低报告展望段与 LLM 预训练泄漏对评测的偏置[^src-time-mmd]。

## 九大领域（数值侧）

| 领域 | 目标变量（示例） | 频率 | 样本规模量级 | 跨度 |
|------|------------------|------|--------------|------|
| Agriculture | Retail Broiler Composite | 月 | ~496 | 1983–Present |
| Climate | Drought Level（多维） | 月 | ~496 | 1983–Present |
| Economy | International Trade Balance | 月 | ~423 | 1989–Present |
| Energy | Gasoline Prices | 周 | ~1479 | 1996–Present |
| Environment | Air Quality Index | 日 | ~11102 | 1982–2023 |
| Health | Influenza patients proportion（US；另含 Africa） | 周 | ~1389 | 1997–Present |
| Security | Disaster and Emergency Grants | 月 | ~297 | 1999–Present |
| Social Good | Unemployment Rate | 月 | ~900 | 1950–Present |
| Traffic | Travel Volume | 月 | ~531 | 1980–Present |

主任务强调**目标变量的单变量预测**并构造对齐文本；部分集提供协变量维以启发后续工作。Health 同时含 US / Africa，用于公平与代表性讨论（Africa 周期更弱、报告更少）[^src-time-mmd]。

## 文本构造流水线

1. **报告源**：每目标 1–2 条高相关、可持续更新的官方报告系列。  
2. **检索源**：2–3 个关键词 + Google API，按周取 top 结果（约 1980–今）。  
3. **Llama3-70B 预处理**：过滤无关、**拆分事实与预测**、摘要；允许 “NA”、要求引用来源以抑幻觉。  
4. **二元时间戳**：人工/周聚合标定 start–end，支持预测、插补、异常等多任务切片[^src-time-mmd]。

统计上：报告 **relevance 高、coverage 低**；检索 **coverage 高、relevance 低**。人工抽检：事实幻觉可滤后约 3/127；误丢相关约 4/52[^src-time-mmd]。

## MM-TSFlib

- 形式：\(g_\theta(X,S)=Y\)；数值 TSF 骨干 ∥ 冻结 LLM + 投影 + pooling，可学习线性融合。  
- 覆盖 **20+** TSF 算法与 **7** 开源 LLM；防泄漏：文本 end ≤ 数值 lookback end。  
- 视界按频率：日 {48,96,192,336}、周 {12,24,36,48}、月 {6,8,10,12}[^src-time-mmd]。

## 关键实证

- **>1000** 实验中约 **95%** 多模态优于对应单模态；MSE 平均降 **>15%**（文内亦述 over 20%），文本富域最高约 **40%**[^src-time-mmd]。  
- 增益与 lookback 内 **相关 fact 数** 正相关；Security 等未来高不确定域受益相对小；短/长视界均稳健[^src-time-mmd]。  
- LLM 骨干规模 / 通用 NLP 能力与 TSF 增益**无明显正相关**；Doc2Vec 弱于 BERT；Time-LLM 式 prompt 前缀对外生文本序列无效甚至有害[^src-time-mmd]。

## 在谱系中的位置

Time-MMD 是**数据与评测基础设施**，不是预测模型。下游常引用为文本协变量基准：

| 工作 | 用法 |
|------|------|
| [[cora-tsfm|CoRA]] / [[unica|UniCA]] | 文本协变量适配；CoRA 报 MSE 0.641 vs UniCA 0.653 |
| [[source-gpt4mts|DP-GPT4MTS]] | Agriculture / Public Health 子集 SOTA |
| [[vot|VoT]]、[[timi|TiMi]]、TaTS 等 | 多域多模态预测 / 插补评测 |
| [[tsfm-covariate-adaptation-comparison]] | 文本模态一行的公共参照 |
| [[time-vlm|Time-VLM]] | 不直接用 Time-MMD 外生文本；内生图文自增强的 VLM 路线，与本基准形成「外生对齐 vs 内生 VLM」对照 |
| [[constrained-text-fusion|CFA / Constrained Fusion]] | 同九域系统证伪 naive add/concat；CFA 低秩 plug-in 约束融合（KDD ’26 MILETS）[^src-constrained-text-fusion] |
| [[cross-modal-misalignment|Cross-modal misalignment]] | 配对 MMCL 理论：文本省略/扰动 = selection/perturbation；表示只保留无偏共享语义——解释为何“对齐文本”质量与覆盖决定可学因子[^src-cross-modal-misalignment] |
| [[pir|PIR]]（Post-forecasting Identification and Revision，Liu et al., NeurIPS 2025） | Energy / Health 子集上验证对齐文本描述作为外生信息的后处理修订（周频，Lin=24）[^src-pir] |

释放元数据：数值（起止、目标、协变量）+ 文本（起止、fact/prediction 内容与来源）。GitHub: `https://github.com/AdityaLab/Time-MMD`[^src-time-mmd]。

## 局限

英语 only；first-cut 融合库；图像/音频未纳入；其他 TSA 任务需再 curation[^src-time-mmd]。

## 相关页面

- [[source-time-mmd]] — 源摘要  
- [[multimodal-time-series-forecasting]] · [[source-cora]] · [[cora-tsfm]] · [[source-unica]] · [[unica]]  
- [[source-gpt4mts]] · [[source-from-news-to-forecast]] · [[source-event-driven-ts-forecasting]] · [[vot]]  
- [[time-vlm]] · [[source-time-vlm]] — 内生 VLM 多模态预测（对照外生文本基准）
- [[constrained-text-fusion]] · [[source-constrained-text-fusion]] — naive 常伤、CFA 受控融合（同九域 >20K 实验）
- [[cross-modal-misalignment]] · [[source-cross-modal-misalignment]] — selection/perturbation 与缓解 vs 利用
- [[tsfm-covariate-adaptation-comparison]] · [[covariate-homogenization]]

[^src-time-mmd]: [[source-time-mmd]]
[^src-constrained-text-fusion]: [[source-constrained-text-fusion]]
[^src-cross-modal-misalignment]: [[source-cross-modal-misalignment]]
[^src-pir]: [[source-pir]]
