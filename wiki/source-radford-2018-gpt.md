---
title: "Improving Language Understanding by Generative Pre-Training（Radford et al., 2018）"
type: source-summary
tags:
  - gpt
  - pre-training
  - language-model
  - nlp
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Improving Language Understanding by Generative Pre-Training（Radford et al., 2018）

OpenAI 技术报告（Preprint, work in progress），GPT-1 的原始出处。作者 Alec Radford、Karthik Narasimhan、Tim Salimans、Ilya Sutskever。源文件：`raw/gpt-1-language-understanding-2018.pdf`。机制页见 [[generative-pretraining]]。

## 问题

NLP 判别任务标注数据稀缺，而深度模型依赖大量标注；从无标注文本利用词级以上信息面临两个未决问题：何种优化目标能学到可迁移表示、迁移时如何避免为各任务修改架构（§1）[^src-radford-2018-gpt]。

## 机制

两阶段半监督框架先在 BooksCorpus（7000+ 本未出版书籍）上以标准语言建模目标训练 12 层 decoder-only Transformer（$d=768$、12 头、FFN 3072、BPE 40,000 merges、GELU、可学习位置嵌入、post-LN，token 困惑度 18.4），再在目标任务上微调，优化 $\mathcal{L}_3 = \mathcal{L}_2 + 0.5\,\mathcal{L}_1$（含辅助 LM 目标）（§3–§4.1）[^src-radford-2018-gpt]。关键设计是任务特定输入变换：将文本对、文档-问题-答案等结构化输入按 traversal 方式拼成单 token 序列（分隔符 `$`，加起止符）；无序句对任务计算两种顺序后将表示相加，多选任务逐答案前向计算后接 softmax，模型架构在任务间几乎不变（§3.3）[^src-radford-2018-gpt]。

## 证据

12 个任务中 9 个超过此前最好成绩（§4.2）：Story Cloze 86.5（+8.9，Table 3）、RACE 59.0（+5.7）、MNLI-m 82.1（+1.5，Table 2）、QNLI +5.8、SciTail +5.0、SNLI 89.9、QQP F1 +4.2、STS-B +1.0、CoLA mc 45.4 对 35.0（Table 4）；GLUE 总分 72.8 对 68.9（Table 4）。消融显示无预训练平均降 14.8 点，Transformer 换单层 LSTM 降 5.6 点；辅助 LM 目标在大数据集有益、小数据集无益；迁移层数越多越好（MultiNLI 全迁移最多 +9，Table 5、Figure 2）（§4.2、§5）[^src-radford-2018-gpt]。

## 边界

RTE（2490 样本）56.0 低于多任务 biLSTM 61.7，作者推测多任务微调有效但未探索；SST-2、MRPC 未超 SOTA（§4.2、Table 4）[^src-radford-2018-gpt]。评测仅覆盖判别式理解任务，无生成任务；引言称 GLUE 绝对提升 5.5%，与 Table 4 两数之差 3.9 点不一致（详见 [[ingest-reports/2026-10-27-gpt|WHY 报告]]数值备注）[^src-radford-2018-gpt]。

## 脚注

[^src-radford-2018-gpt]: [[source-radford-2018-gpt]]
