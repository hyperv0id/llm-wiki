---
title: "What Is a Looped Transformer? Complete Guide to Recurrent Depth and OpenAI's Astra（Tosea.ai 博客）"
type: source-summary
tags:
  - recurrent-depth
  - latent-reasoning
  - monitorability
  - secondary-source
created: 2026-10-25
last_updated: 2026-10-25
source_count: 2
confidence: low
status: active
---

# What Is a Looped Transformer?（Tosea.ai 博客，2026-09-02）

厂商博客（Tosea.ai，做文档转 PPT 的产品），15 分钟阅读量级，无同行评审、无实验。本页只作背景：它把 [[recurrent-depth|循环深度]] 的定义、公开研究证据与 2026-09 关于 OpenAI「Astra」的报道串在一起[^src-tosea-looped-transformer]。

## 定义与主张

博客把 looped transformer 定义为「用单个共享块替换部分层栈、反复施加」，并给出两个推论：算力变成运行时可调的旋钮；推理不再必然落成文字。它明确把这两点分成「吸引力」与「争议」两件事，前者是效率，后者是可监督性[^src-tosea-looped-transformer]。

对 Huginn（arXiv:2502.05171）的复述与方法一致：prelude/core/coda 三段、core 每轮拼接重注入输入嵌入、初值随机、两处设计服务于 path independence；并把论文 §9 的 MoE 对照原样引出——循环深度擅长学推理模式、MoE 擅长存储检索[^src-tosea-looped-transformer]。

## 报道性内容（未独立核实）

- The Information 2026-09-01 报道 OpenAI 的 Astra 使用 recurrent depth；OpenAI 未确认也未否认该架构说法[^src-tosea-looped-transformer]。
- OpenAI 首席科学家 2026-09-02 回应称当前前沿模型（含 Astra）的计算图深度「在 GPT-4 的两倍以内」，博客把这条标为「已确认、测量未公开」[^src-tosea-looped-transformer]。
- 2025-07 的 CoT monitorability 立场论文在 "Novel architectures" 一节点名 Coconut 与循环深度一线，认为 latent 推理可能失去 CoT 的安全优势；博客强调该文作者含 OpenAI 首席科学家本人[^src-tosea-looped-transformer]。
- 谱系表：Universal Transformers 2018 → ALBERT 2019 跨层共享 → Coconut 2024 → Huginn 2025 → Mixture-of-Recursions 2025 → Ouro/LoopLM 2025（学习退出）[^src-tosea-looped-transformer]。

以上均只有这一手二手来源，本 wiki 不据此对任何厂商架构下事实断言。

## 与论文的数字核对

博客复述的主表数值与 arXiv:2502.05171 基本一致（ARC-E 69.91、MMLU 31.38、HumanEval 23.17 vs OLMo-7B 12.80、OLMo-2 的 82.79/60.56、Table 4 的 53.62 vs 46.42 与 9.02 vs 1.82）[^src-tosea-looped-transformer][^src-2502-05171-recurrent-depth]。三处偏差：

| 博客写法 | 论文原文 |
|---|---|
| 「test-time loop scaling to roughly 50B **materialized parameters**」 | 50B 是 FLOP 预算等价；Fig 1 的材料化参数量在 $r=64$ 时为 103B[^src-2502-05171-recurrent-depth] |
| $r=1$ 行 SciQ 47.19 | 47.10（Table 4 的 0.8T 行）[^src-2502-05171-recurrent-depth] |
| $r=1$ 行 GSM8K CoT 0.00/0.15 | 0.00/0.00；0.00/0.15 属 180B 早期 checkpoint 的 $r=1$ 行[^src-2502-05171-recurrent-depth] |

[^src-tosea-looped-transformer]: [[source-tosea-looped-transformer]]
[^src-2502-05171-recurrent-depth]: [[source-2502-05171-recurrent-depth]]
