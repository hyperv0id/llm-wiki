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

**Authors**: Ming Jin（Griffith）、Yaxuan Kong（Oxford）、Yuxuan Liang（HKUST-GZ）等 16 人；通讯 Shirui Pan（Griffith）、Qingsong Wen（Squirrel Ai Learning）
**Venue**: arXiv:2310.10196（v1 推断为 2023-10，据 arXiv 编号 2310；v3 [cs.LG] 8 Jun 2026，ACM 投稿稿）

## 问题与定位

时序与时空数据上的大模型工作激增，但论文指出既有综述 "typically focus on a single modality or model family and rarely provide an integrated view"（L240-242）。Table 1 中 11 篇对比综述（Madan 2024、Capone 2025、Wang 2025、Kottapalli 2025、Shi 2025、Liang 2025 等）无一在四模态列全勾；本综述为表中唯一全勾行（按表格勾选计，非论文自述，原文措辞为 "an integrated temporal-data perspective"，L243-244）。原文无 first/pioneer 表述，仅可据 v1 首发（2023-10）推断其属早期系统综述。

## 分类体系

沿四个维度组织：data categories、model architectures、model scopes、application domains/tasks（L750-751）。顶层二分 LM4TS/LM4STD；LM4TS 再分 LLM4TS（复用语言中心大模型，冻结或微调均可）与 PFM4TS（为时序模态自建 backbone），作者说明该区分 "practical rather than ontological"（L757-759）。LM4STD 聚焦三个模态：spatio-temporal graphs、temporal knowledge graphs、videos（L862-863）；TKG 把三元组 (s,p,o) 扩展为带时间戳四元组 (s,p,o,t)（L1663）。

## 证据与代表工作

Table 2 收录 42 个 TS 方法（25 LLM4TS + 17 PFM4TS）与 45 个 ST 方法，合计 87。论文提出 LLM4TS 四条路线：prompting、tokenization、decomposition、reprogramming（L897-898）——PromptCast 提出 prompt-based forecasting 并发布 PISA 数据集（L899-902），LLMTime 证明 LLM 是 zero-shot 时序学习器（L903-904），TEMPO 引入分解与 soft prompts（L904-905），Time-LLM 做 reprogramming（ICLR 2024，L906-907）。PFM4TS 代表：PatchTST（ICLR 2023）、TimesFM、MOIRAI、Chronos、Timer（ICML 2024）、Lag-llama、MOMENT、Time-MoE、Chronos-2；STG 线有 STG-LLM、UrbanGPT（KDD 2024）；气候 PFM 有 Pangu-Weather、ClimaX。第 5.4 节提炼三大共性挑战：tokenization 为模型可读单元、有限 context 预算下的长程依赖、外部知识与模态对齐（L1799-1808）。

## 未来方向（Sec 7，原文明言 top six）

1) 迁移的理论分析——原文承认 tokenization/注意力何时有效或因分布偏移、数值精度损失失效无理论答案（L2155-2158）；2) 多模态与多分辨率对齐（L2164-2169）；3) 持续学习与无遗忘适配（L2175-2178）；4) 可解释性，含反事实与因果推理（L2184-2198）；5) 隐私与对抗攻击（差分隐私、联邦学习，L2201-2205）；6) LLM agents 时序决策（OOD 检测与弃权、safe exploration，L2208-2217）。

## 局限

论文自认 PFM4TS 方向 "relatively nascent"，收录模型未必满足 Sec 2.2 定义的通用 PFM 标准（L760-762）；LM4TS 简单可扩展但可能丢失细粒度数值信息，TKG 方法受事实不完备与稀疏制约，视频模型 token 与内存开销高（L1808-1811）。Zeng et al. "Are Transformers Effective for Time Series Forecasting?"（AAAI 2023，[272] L2864）质疑的是 Transformer 架构本身，不应外推为 LLM4TS 负结果。

## 与 wiki 其他综述比较

- [[source-st-foundation-models-survey]]：专注 ST 基础模型实证，本综述更早覆盖双线。
- [[source-stfm-pipeline-review]]：pipeline 视角 vs 分类学视角。
- [[source-deep-time-series-survey]]：仅覆盖 TS 架构模型与基准。
- [[source-multimodal-ts-survey]]：多模态 TS 专述，垂直化未来方向 2。

[^src-large-models-ts-st-survey]: [[source-large-models-ts-st-survey]]
