---
title: "Re-Reading Improves Reasoning in Large Language Models (Re2, EMNLP 2024)"
type: source-summary
tags:
  - prompting
  - reasoning
  - re-reading
  - input-encoding
  - emnlp-2024
created: 2026-09-13
last_updated: 2026-09-13
source_count: 2
confidence: medium
status: active
---

# Re2（Re-Reading Improves Reasoning, EMNLP 2024）

## 摘要
Re2 由 Xiaohan Xu 等 8 人提出，发表于 EMNLP 2024 主会。[^src-re-reading-improves-reasoning]

论文关注输入编码：causal attention 使问题中较早的词不能利用后置焦点形成表示；作者认为既有提示研究更偏重输出侧的思维激发。[^src-re-reading-improves-reasoning]

机制：在单条提示内把完整问题写两遍，第二份以 Read the question again: 引导，后接思维激发句。[^src-re-reading-improves-reasoning]作者把第一份为第二份提供后文信息的效果称作“双向”编码，并给出 LLaMA-2 注意力热图；这不改变 causal mask。[^src-re-reading-improves-reasoning]ChatGPT/GSM8K 的 Table 7 报告，显式重读指引优于裸重复，而重复思维激发句本身未带来明显收益。[^src-re-reading-improves-reasoning]

证据：14 个算术、常识与符号推理数据集、112 组实验，模型含 davinci-003、ChatGPT、LLaMA-2-13B/70B；作者报告除 vanilla ChatGPT 少数设置外，重读版相对同模型原提示有提升，并检验零/少样本、自洽性与思维链组合。[^src-re-reading-improves-reasoning]

局限（作者自述）：以实证为主、缺理论分析；问题加倍增加输入长度，长问题推理效率有所下降。[^src-re-reading-improves-reasoning]
## 来源与阅读范围

- 出版页：https://aclanthology.org/2024.emnlp-main.871/
- PDF：https://aclanthology.org/2024.emnlp-main.871.pdf （DOI: 10.18653/v1/2024.emnlp-main.871，pp. 15549–15575）
- 本地锁定版本：`downloads/re-reading-improves-reasoning.pdf`（未放入 raw/）
- 阅读范围：pdftotext 全文提取后，完整阅读摘要、§1 引言、§2 方法（re2 提示模板与式 1–3）、§3.1 基准列表、§3.3–3.5 结果讨论与指令变体分析（Table 7）、推理效率分析、Limitations；各数据集的完整数值表与附录提示模板为浏览级。

## 与主论文的关系

[[trace-as-state]] §2 将 Re2 引为"无先验推理文本的 direct rereading baseline"；其 Table 3 的 Re2 行标注 [x, q, x, q]，重复对象是长上下文加 question，与原作"只重复 question"的设置不同，差异核查见 [[trace-as-state-positioning]] 。[^src-trace-as-state]

[^src-re-reading-improves-reasoning]: [[source-re-reading-improves-reasoning]]
[^src-trace-as-state]: [[source-trace-as-state]]
