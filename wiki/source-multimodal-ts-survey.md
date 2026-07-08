---
title: "Multi-modal Time Series Analysis: A Tutorial and Survey"
type: source-summary
tags:
  - survey
  - multimodal
  - time-series
  - benchmarks
  - 2025
created: 2026-07-07
last_updated: 2026-07-08
source_count: 1
confidence: medium
status: active
---

# Multi-modal Time Series Analysis: A Tutorial and Survey

> Yushan Jiang\*, Kanghui Ning\*, Zijie Pan\*, Xuyang Shen, Jingchao Ni, Wenchao Yu, Anderson Schneider, Haifeng Chen, Yuriy Nevmyvaka, Dongjin Song (University of Connecticut, Morgan Stanley, NEC Laboratories America). arXiv:2503.13709v1, Mar 2025.

本文系统地综述了多模态时间序列分析方法，提出了统一的跨模态交互框架，将现有方法分类为**融合（Fusion）、对齐（Alignment）和迁移（Transference）**三种交互类型，涵盖输入、中间表示和输出三个层面。[^src-multimodal-ts-survey]

---

## 核心挑战

多模态时间序列分析面临三大核心挑战：[^src-multimodal-ts-survey]
- **数据异质性（Heterogeneity）**：不同模态具有不同的统计属性、结构和维度——时序数据是顺序的、带时间依赖的，而文本和图像携带丰富的上下文语义。
- **模态差距（Modality Gap）**：不同模态的语义空间不一致，对齐到统一表示空间极具挑战。
- **时间错位（Temporal Misalignment）**：文本、表格或视觉上下文可能出现在不同的时间步或粒度上，阻碍有意义的跨模态交互。
- **噪声与冗余（Noise & Redundancy）**：现实数据不可避免地包含噪声和无关信息（如金融新闻中的投机性叙述），可能误导相关性学习。

## 统一跨模态交互框架

### 融合（Fusion）
- **输入级融合**：将时间序列、表格数据、文本整合为统一文本提示，通过指令微调或零样本推理查询 LLM。[^src-multimodal-ts-survey]
- **中间级融合**：个体模态编码器先将数据映射到共享潜空间，再通过加法或拼接组合表示。代表性方法包括 Time-MMD、GPT4MTS、Time-LLM 等。[^src-multimodal-ts-survey]
- **输出级融合**：不同模态分别贡献最终预测，可通过 MLP 离线合成或 LLM agent 协同推理。代表工作为 MOAT、TimeCAP。[^src-multimodal-ts-survey]

### 对齐（Alignment）
- **输入对齐**：数据预处理层面的时间对齐，处理缺失值、不规则采样和不同粒度。依赖领域知识。[^src-multimodal-ts-survey]
- **中间对齐**：包括自注意力（self-attention，多模态间的联合对齐）、交叉注意力（cross-attention，以时序为 query 来对齐文本/图像上下文）和门控机制（gating，显式控制模态贡献）。对比学习也被用于对齐跨模态表示。[^src-multimodal-ts-survey]
- **组件输出对齐**：LLM agent 通过迭代自我反思和检索增强（如 TimeCAP 的上下文检索、MATMCD 的因果约束推理）来精炼跨模态对齐。[^src-multimodal-ts-survey]

### 迁移（Transference）
- **输入级迁移**：通过元信息描述（meta-description）或 LLM 生成细粒度文本来增强时间序列；或将时序转为图像（feature imaging）或表格形式进行模态转换。[^src-multimodal-ts-survey]
- **中间/输出级迁移**：更面向任务——如 EEG 转文本、文本检索时序等端到端跨模态生成。[^src-multimodal-ts-survey]

## 数据集与基准

综述系统整理了多领域多模态时序数据集：[^src-multimodal-ts-survey]
- **Healthcare**：MIMIC-III/IV（TS+Text+Tabular）、ICBHI（TS+Text）、Coswara、PTB-XL、ZuCo、Image-EEG
- **Finance**：FNSPID（已覆盖 4000+ 公司 1999–2023）、ACL18、CIKM18、DOW30
- **Multi-domain**：Time-MMD（9 领域）、TimeCAP（3 领域）、NewsForecast（多领域）、TTC（气候+医疗）、CiK（7 领域）、TSQA（12 领域、200k QA 对）
- **Retail**：VISUELLE（TS+Image+Text）
- **IoT**：LEMMA-RCA（TS+Text）
- **Speech**：LRS3、VoxCeleb2
- **Traffic**：NYC-taxi/bike（ST+Text）
- **Environment**：Terra（ST+Text，45 年全球数据）

## 未来研究方向

- **多模态时序推理**：结合 RAG 系统和 LLM 推理模型（CoT、ToT），构建统一推理框架。[^src-multimodal-ts-survey]
- **决策制定**：利用多模态上下文的预测信号和解释，开发自适应、可解释、可靠决策支持系统。[^src-multimodal-ts-survey]
- **领域泛化**：应对分布偏移——不仅来自时间序列，也来自其他模态。需要保留跨模态领域不变成分。[^src-multimodal-ts-survey]
- **缺失与噪声模态的鲁棒性**：模态特异性插补、降噪和相关量化。[^src-multimodal-ts-survey]
- **伦理与偏见**：公平性约束、反事实分析和对抗去偏。[^src-multimodal-ts-survey]

## 交叉链接

- [[source-terra]] — Terra 数据集，该综述收录的环境领域代表性数据集
- [[source-stfm-pipeline-review]] — Pipeline 视角的 STFM 综述，与本综述在多模态数据处理上互补
- [[source-timecap]] — TimeCAP，综述中输出级融合和对齐的代表方法（AAAI 2025 Oral）
- [[source-time-llm]] — Time-LLM，综述中中间级融合的编/解码架构（ICLR 2024）


[^src-multimodal-ts-survey]: [[source-multimodal-ts-survey]]
