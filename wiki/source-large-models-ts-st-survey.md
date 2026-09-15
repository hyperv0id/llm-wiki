---
title: "Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook"
type: source-summary
tags:
  - survey
  - time-series
  - spatio-temporal
  - llm
  - foundation-model
  - taxonomy
created: 2026-09-15
last_updated: 2026-09-15
source_count: 1
confidence: medium
status: active
---

# Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook

**Authors**: Ming Jin (Griffith), Yaxuan Kong (Oxford), Yuxuan Liang (HKUST-GZ), Chaoli Zhang, Siqiao Xue (Ant), Xue Wang (Alibaba) 等
**Venue**: arXiv:2310.10196（首版 2023-10；v3 于 2026-06 修订，ACM 投稿稿，Table 1 定位标注 2025）

## 历史地位

首版于 2023 年 10 月，与 PromptCast、LLMTime、Time-LLM 的第一波 LLM4TS 探索同期，早于 TimesFM/Chronos/MOIRAI（2024）等 time series foundation model 落地潮，是该领域最早系统综述之一；其 LM4TS/LM4STD、LLM4TS/PFM4TS 命名成为后续文献通用术语[^src-large-models-ts-st-survey]。综述自述对比十余篇相关综述（Madan et al. 2023、Capone et al. 2023、Wang et al. 2024、Kottapalli et al. 2024、Shi et al. 2025、Liang et al. 2025），并声称是唯一同时覆盖 time series 与 spatio-temporal 两条线的[^src-large-models-ts-st-survey]。

## 分类体系

按四维组织：数据类别（time series vs spatio-temporal）、模型架构（LLM vs PFM）、模型范围（general-purpose vs domain-specific：交通/金融/医疗）、应用任务[^src-large-models-ts-st-survey]。顶层二分 LM4TS / LM4STD；LM4STD 按模态再分三类：spatio-temporal graphs、temporal knowledge graphs（三元组 (s,p,o) 扩展为带时间戳四元组 (s,p,o,t)）、视频[^src-large-models-ts-st-survey]。Table 2 收录 43 个 TS 方法（24 LLM4TS + 19 PFM4TS）。作者明确说明 LLM/PFM 之分是 practical 而非 ontological——前者复用语言中心基础模型，后者为时序模态自建 backbone[^src-large-models-ts-st-survey]。

## 机制与代表模型

LLM4TS 四条技术路线：prompting（PromptCast 提出 prompt-based forecasting 任务并发布 PISA 指令数据集）、tokenization（LLMTime 零样本）、decomposition + soft prompts（TEMPO）、reprogramming（Time-LLM，ICLR 2024）[^src-large-models-ts-st-survey]。多任务扩展有 Time-MQA、ChatTS、Time-MMD；PFM4TS 代表：PatchTST（2023）、TimesFM（2024）、MOIRAI、Chronos、Lag-llama、Timer、MOMENT、Time-MoE、Chronos-2[^src-large-models-ts-st-survey]。LM4STD 中 STG 线有 STG-LLM（STG-Tokenizer + STG-Adapter）、UrbanGPT（2024）、RePST；TKG 线有 Chain-of-History、LLM-DA、LATE、G2S、TeRDy；视频线有 Video-ChatGPT、Video-LLaMA、MovieChat、Video-XL；气候 PFM 有 Pangu-Weather、ClimaX、FourCastNet-3[^src-large-models-ts-st-survey]。第 5.4 节提炼两大阵营三大共性挑战：tokenization 为模型可读单元、有限 context 预算下的长程依赖、外部知识与模态对齐[^src-large-models-ts-st-survey]。

## 未来方向（6 个）与后续验证

1) 大模型迁移到时序的理论分析；2) 多模态与多分辨率对齐——后续 [[source-multimodal-ts-survey]]（2025）、TaTS（ICLR 2026）沿此展开；3) 持续学习与无遗忘适配——EAC（ICLR 2025）prompt pool 可视为回应；4) 可解释性（反事实/因果推理）；5) 隐私与对抗攻击（差分隐私、联邦学习）；6) LLM agents 时序决策（OOD 检测、弃权、safe exploration）——TimeCAP（AAAI 2025）已初步验证[^src-large-models-ts-st-survey]。

## 局限

作者自认 PFM4TS 方向 nascent，收录模型未必满足其自定义的通用 PFM 标准[^src-large-models-ts-st-survey]；第 7.1 节承认语言 tokenization 与注意力机制何时有效、何时因分布偏移/数值精度损失失效仍无理论答案；TKG 部分受制于事实不全、时间戳稀疏；对 LLM4TS 后续负结果（如 "Are Transformers Effective for Time Series Forecasting?" 一类质疑）未作重点处理[^src-large-models-ts-st-survey]。

## 与 wiki 其他综述比较

- [[source-st-foundation-models-survey]]（A*STAR, 2025）：只聚焦 STFM，4 泛化能力维度 + 6 模型实证；本综述覆盖 TS+ST 双线且含 LLM 复用路线，是它的先声。
- [[source-stfm-pipeline-review]]（2025）：按 pipeline（数据 harmonization→模型设计→适配）视角重审 ST 基础模型；本综述按分类学视角。
- [[source-deep-time-series-survey]]（TSLib）：只覆盖 TS 架构模型与基准，无 LLM/ST 维度。
- [[source-multimodal-ts-survey]]（2025）：专注多模态 TS 的融合/对齐/迁移三分法，是本综述未来方向 2 的垂直化。

[^src-large-models-ts-st-survey]: [[source-large-models-ts-st-survey]]
