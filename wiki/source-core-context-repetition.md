---
title: "CoRe: Unleashing Multi-Hop Reasoning through Repetition of Misordered Context (Findings NAACL 2025)"
type: source-summary
tags:
  - prompting
  - multi-hop-qa
  - context-order
  - context-repetition
  - naacl-2025
created: 2026-09-13
last_updated: 2026-09-13
source_count: 2
confidence: medium
status: active
---

# CoRe（Context Repetition, Findings NAACL 2025）

## 摘要
CoRe 由 Sangwon Yu 等 6 人提出，发表于 Findings of NAACL 2025。[^src-core-context-repetition]

论文关注多跳问答的 misordered context：支持文档相对顺序也会影响推理；作者报告在无噪声上下文中，仅换文档顺序即可产生最高 26 个百分点的 F1 差。[^src-core-context-repetition]

定理 1 证明，把含 k 个支持文档的上下文重复 k 次，可让任意文档排列作为有序子序列出现；从第 i 份中取相应文档，其余内容视作间隔噪声。[^src-core-context-repetition]这保证顺序存在，而非真实模型必然利用；实践用预设重复数 k̂（实验 1–3），通过聊天模板复述上下文，默认使用 assistant 角色，不需训练。[^src-core-context-repetition]

证据覆盖 HotpotQA、2WikiMultihopQA、MuSiQue 与逆序链表合成任务；作者报告，Llama-3.1-8B-Instruct 在 2Wiki 上相对不重复上下文的 F1 最高增加 30 个百分点，合成任务 accuracy 最高增加 70 个百分点，并检验与检索式思维链的组合。[^src-core-context-repetition]

局限（作者自述）：重复增加内存与时间成本，也重复了噪声；后者可能损害推理，只重复核心内容被列为未来方向。[^src-core-context-repetition]
## 来源与阅读范围

- 出版页：https://aclanthology.org/2025.findings-naacl.360/
- PDF：https://aclanthology.org/2025.findings-naacl.360.pdf （DOI: 10.18653/v1/2025.findings-naacl.360，pp. 6450–6470）
- 本地锁定版本：`downloads/core-context-repetition.pdf`（未放入 raw/）
- 阅读范围：pdftotext 全文提取后，完整阅读摘要、§1 引言、§2 相关工作、§3 方法（定义 1–3、定理 1、推论 1.1、§3.4 实践方法）、§4 实验设置与 §4.3 分析、Limitations、附录提示模板节选；主结果数值表为浏览级。

## 与主论文的关系

[[trace-as-state]] §2 转述 CoRe "repeating the full context" 降低文档顺序敏感性，与原作机制一致；主论文未在 Table 3 或任何消融中与 CoRe 对照，无 head-to-head 数据（见 [[trace-as-state-positioning]]）。[^src-trace-as-state]

[^src-core-context-repetition]: [[source-core-context-repetition]]
[^src-trace-as-state]: [[source-trace-as-state]]
