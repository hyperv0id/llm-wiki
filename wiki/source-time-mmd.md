---
title: "Time-MMD: Multi-Domain Multimodal Dataset for Time Series Analysis"
type: source-summary
tags:
  - multimodal-time-series
  - dataset
  - time-series-forecasting
  - text-numeric-alignment
  - neurips-2024
created: 2026-07-28
last_updated: 2026-08-06
source_count: 2
confidence: high
status: active
---

# Time-MMD 源文件摘要

**来源**: Haoxin Liu, Shangqing Xu, Zhiyuan Zhao, Lingkai Kong, Harshavardhan Kamarthi, Aditya B. Sasanur, Megha Sharma, Jiaming Cui, Qingsong Wen, Chao Zhang, B. Aditya Prakash. *Time-MMD: Multi-Domain Multimodal Dataset for Time Series Analysis.* NeurIPS 2024 Datasets and Benchmarks Track. arXiv:2406.08627v4 (24 Jan 2025). Georgia Institute of Technology / Squirrel AI. Code & data: `https://github.com/AdityaLab/Time-MMD`. raw: `raw/time-mmd-multi-domain-multimodal-dataset-for-time-series-analysis.pdf`[^src-time-mmd]

## 核心论点

真实世界时间序列分析（TSA）几乎总是**多模态**的：流行病学家把感染曲线与政策/报告并读，能源分析师把价格与市场评论并读。然而主流 TSA 模型长期停留在**单模态数值**。LLM 时代的若干工作引入了内生文本（把统计量写成自然语言、reprogramming 进 LLM），但**外生文本**——与数值时间步对齐的事件、政策与新闻——仍因缺少高质量公共数据而难以系统研究。作者指出既有多模态时序集的三大缺口：**领域过窄**（多集中于金融股价）、**对齐过粗**（同域新闻堆到单只股票上，噪声巨大）、**评测污染**（报告常含展望段；测试文本可能已进入 LLM 预训练语料）。Time-MMD 的回答是：构建首个**跨 9 主域**、**细粒度目标对齐**、**显式控污染**的数值–文本时序数据集，并用 **MM-TSFlib** 做 first-cut 多模态预测试点，证明文本+数值相对纯数值可稳定增益[^src-time-mmd]。

## 数据集构造

数值侧优先政府与可验证、可更新源；九域为 Agriculture（broiler 复合价）、Climate（干旱）、Economy（国际贸易差额）、Energy（汽油价）、Environment（空气质量）、Health（流感比例，含 US 与 Africa 公平对照）、Security（灾害应急拨款）、Social Good（失业率）、Traffic（出行量）；频率覆盖日/周/月，多数更新至 **2024-05**，跨度可至 1950。文本侧组合两类源：**精选报告系列**（高相关、覆盖有限）与 **关键词检索**（覆盖广、冗余高）。Llama3-70B 流水线过滤无关内容、将内容**拆分为事实与预测**并摘要（允许 NA、要求引用来源以抑幻觉）；**二元起止时间戳**标定有效期，使同一语料可切片服务预测及潜在插补/异常任务。统计上报告 relevance 高、coverage 低，检索相反；人工抽检：可滤后事实幻觉约 3/127，误丢相关约 4/52[^src-time-mmd]。

## MM-TSFlib 与实验发现

多模态 TSF 形式化为 \(g_\theta: X \times S \to Y\)。库在 TSlib 上把 **20+** 单模态骨干与 **7** 开源 LLM（BERT、GPT-2 系列、Llama-2/3 等）接到统一管线：数值与文本分路建模，冻结 LLM、只训投影与 pooling，可学习线性加权融合；并强制输入文本 end 不晚于数值 lookback end 以防泄漏。在九域、短–长视界上开展 **>1000** 次实验：约 **95%** 设置上多模态优于对应单模态；MSE 平均降低 **超过 15%**（引言亦写 over 20%），文本丰富域最高约 **40%**。增益与 lookback 内相关 fact 密度正相关，对视界长度较稳健；Security 等未来高不确定域受益相对较小。更换 LLM 骨干时，自然语言能力/参数规模与 TSF 增益**无清晰正相关**；从零训练的 Doc2Vec 可用但整体弱于 BERT；Time-LLM 式 “prompt 前缀” 不适配外生文本序列，常劣于纯数值基线——说明需要专门的序列文本融合，而非静态任务提示[^src-time-mmd]。

## 局限与谱系位置

局限包括：文本仅为英语；融合框架是投影加权 first-cut 而非最优架构；未纳入图像/音频；其它 TSA 任务仍需额外 curation。展望覆盖多模态插补、异常检测与多模态基础模型。作为**数据与评测基础设施**，Time-MMD 已成为 [[cora-tsfm|CoRA]]、[[unica|UniCA]]、[[source-gpt4mts|DP-GPT4MTS]]、[[vot|VoT]]、[[timi|TiMi]]、TaTS 等文本协变量/多模态工作的公共参照；实体细节见 [[time-mmd]][^src-time-mmd]。PIR（Liu et al., NeurIPS 2025）在 Energy / Health 子集（周频，1996/1997–2024-05）上用对齐文本描述作为外生信息，验证其后处理修订框架在文本协变量场景下的泛化性[^src-pir]。

## 相关页面

- [[time-mmd]] — 数据集实体
- [[multimodal-time-series-forecasting]] · [[source-cora]] · [[cora-tsfm]] · [[source-gpt4mts]] · [[source-from-news-to-forecast]] · [[source-event-driven-ts-forecasting]] · [[vot]] · [[tsfm-covariate-adaptation-comparison]]

[^src-time-mmd]: [[source-time-mmd]]
[^src-pir]: [[source-pir]]
