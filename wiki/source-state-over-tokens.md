---
title: "State over Tokens: Characterizing the Role of Reasoning Tokens (arXiv 2512.12777)"
type: source-summary
tags:
  - reasoning-tokens
  - conceptual-framework
  - interpretability
  - position-paper
  - chain-of-thought
created: 2026-09-13
last_updated: 2026-09-13
source_count: 2
confidence: medium
status: active
---

# State over Tokens（SoT, arXiv:2512.12777）

## 摘要
State over Tokens（SoT）由 Mosh Levy 等 4 人撰写，是 2025-12-14 提交的约 7 页概念论文，不报告作者自己的实验。[^src-state-over-tokens]

作者综述指出，reasoning tokens 可有计算作用，却不应默认是忠实解释：文本可能不完整、省略影响因素，或与模型实际计算语义错位（§2.1）。[^src-state-over-tokens]

框架把自回归生成写为纯函数的递归应用；在该设定中，token 序列是跨无状态计算周期持久的信息载体，KV cache 不携带超出可由前缀重算的信息。[^src-state-over-tokens]作者据此建议把推理前缀读作外化计算状态，而非完整语言叙事。[^src-state-over-tokens]白板类比与卡特兰数例用于说明：只需外部化下一步所需的信息，同一序列对人是文本、对模型承担状态功能，二者不必具有相同语义。[^src-state-over-tokens]

论文提出的开放问题包括如何解码这些状态、自然语言是否是特殊计算介质，以及计算状态能否同时承担忠实解释；其论点属于概念框架，不是状态可恢复性的实验证明。[^src-state-over-tokens]
## 来源与阅读范围

- arXiv 摘要页：https://arxiv.org/abs/2512.12777
- PDF：https://arxiv.org/pdf/2512.12777 （v1，2025-12-14）
- 本地锁定版本：`downloads/state-over-tokens.pdf`（未放入 raw/）
- 阅读范围：pdftotext 全文提取后，正文 §1–§7（含摘要、白板类比、形式框架、两个误解、本体论分歧、开放问题、结论）完整读完；参考文献列表为浏览级。

## 与主论文的关系

[[trace-as-state]] §2 引用 SoT 描述"growing reasoning prefix as externalized computational state"，并据此把 reasoning trace 视作 task state 的不完美 textual proxy；SoT 本身不做任何输入顺序或位置干预，也不讨论跨 pass 复用。把主论文的干预概括为"将 SoT 式状态观操作化为跨 pass 的位置干预"是本 wiki 的归纳，主论文仅以相容性措辞引用 [INFERENCE]（见 [[trace-as-state-positioning]]）。[^src-trace-as-state]

[^src-state-over-tokens]: [[source-state-over-tokens]]
[^src-trace-as-state]: [[source-trace-as-state]]
