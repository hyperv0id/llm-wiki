---
title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding（Devlin et al., 2019）"
type: source-summary
tags:
  - bert
  - pre-training
  - masked-language-model
  - transformer
  - nlp
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding（Devlin et al., 2019）

NAACL 2019，arXiv:1810.04805v2（2019-05-24），Google AI Language。作者 Jacob Devlin、Ming-Wei Chang、Kenton Lee、Kristina Toutanova。源文件：`raw/bert-1810-04805.pdf`。机制页见 [[bert]]。

## 问题

预训练语言表示迁移到下游任务有 feature-based（ELMo）与 fine-tuning（OpenAI GPT）两条路线，但预训练目标均为单向语言模型；单向限制了可选架构，对句级任务次优，对问答这类需要双侧上下文的 token 级任务尤其不利（§1）[^src-devlin-2019-bert]。

## 机制

用两个无监督任务预训练多层双向 Transformer encoder。MLM 随机遮住 15% 的 WordPiece，被选位置按 80% `[MASK]`、10% 随机词、10% 原词替换，只预测被遮位置的原 token；NSP 以 50/50 构造相邻句对与随机句对，用 `[CLS]` 的二分类头预测。输入表示是 token、段（A/B）与位置三种嵌入之和；BASE 为 12 层/768/12 头/110M，LARGE 为 24 层/1024/16 头/340M。预训练语料为 BooksCorpus 800M 词加英文 Wikipedia 2,500M 词，batch 128,000 token、1M 步。微调只加一个输出层并端到端训练全部参数，句对、单句、抽取式问答与多选共用同一骨干（§3–§4）[^src-devlin-2019-bert]。

## 证据

GLUE test 平均分 BASE 79.6、LARGE 82.1，OpenAI GPT 75.1（Table 1）；官方榜 LARGE 80.5 对 GPT 72.8（§4.1）。SQuAD 1.1 test F1 93.2（7 模型集成 + TriviaQA），SQuAD 2.0 test F1 83.1（Table 2–3）；SWAG test 86.3（Table 4）。消融实验显示：去除 NSP 使 QNLI 88.4→84.9；改为从左到右且无 NSP 后 SQuAD 88.5→77.8、MRPC 86.7→77.5；模型由 3 层/768 增至 24 层/1024，MNLI-m 77.9→86.6（Table 5–6）。固定特征（末四层拼接）做 NER 得 96.1 dev F1，仅比全参数微调低 0.3（Table 7）（§5）[^src-devlin-2019-bert]。

## 边界

`[MASK]` 在微调阶段不出现，导致预训练与微调存在不匹配，80/10/10 仅起缓解作用（§3.1）[^src-devlin-2019-bert]。每 batch 仅监督 15% 的 token，MLM 收敛慢于从左到右 LM，需 1M 步预训练（C.1）[^src-devlin-2019-bert]。GLUE 仅报告单任务微调并排除 WNLI，Table 1 平均分与官方榜分口径不同（Table 1 题注、B.1）[^src-devlin-2019-bert]。预训练语料仅限英文，未做跨语言评测（§3.1）[^src-devlin-2019-bert]。

## 脚注

[^src-devlin-2019-bert]: [[source-devlin-2019-bert]]
