---
title: "Language Models are Unsupervised Multitask Learners (GPT-2)"
type: source-summary
tags:
  - language-model
  - transformer
  - zero-shot
  - pre-ln
  - openai
created: 2026-10-26
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# Language Models are Unsupervised Multitask Learners（GPT-2）

**Radford, Wu, Child, Luan, Amodei & Sutskever（OpenAI），2019 技术报告，非同行评议（PDF metadata 的 ICML 著录为模板残留，正文无会议标识）。全文 `raw/gpt2-language-models-are-unsupervised-multitask-learners.pdf`，官方地址 https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf**

## 问题

自然语言处理传统上按任务收集标注数据、训练专用模型，导致模型成为窄域专家而非通才，且对数据分布与任务表述的微小改变敏感（§1）[^src-gpt2-2019-language-models]。作者将此归因于单任务、单领域训练，主张在足够大且多样的语料上训练语言模型，模型能以无监督方式学会任务，并将任务条件写进建模目标：$p(\text{output}\mid\text{input},\text{task})$（§2）[^src-gpt2-2019-language-models]。

## 方法

训练语料 WebText 取 Reddit 上 karma $\geq 3$ 的外链文本，去重清洗后略多于 800 万篇、共 40 GB，并移除全部 Wikipedia 文档（§2.1）[^src-gpt2-2019-language-models]。输入表示采用 byte-level BPE：基础词表 256，且禁止跨字符类别合并，从而对任意 Unicode 串均可赋予概率（§2.2）[^src-gpt2-2019-language-models]。模型共有 117M/345M/762M/1542M 四个尺寸，最大者 GPT-2 为 48 层、$d_{model}=1600$（Table 2）[^src-gpt2-2019-language-models]。结构沿用 GPT（Radford et al., 2018），共有四处改动：LayerNorm 移至各子块输入端、最后一个 self-attention block 后追加一层 LayerNorm、残差层权重初始化按 $1/\sqrt{N}$ 缩放（$N$ 为残差层数）、词表扩至 50,257；上下文从 512 增至 1024，batch size 为 512（§2.3）[^src-gpt2-2019-language-models]。详见 [[pre-ln-transformer|Pre-LN Transformer]]。

## 证据

在零样本、无微调测试中（Table 3），8 个语言建模数据集有 7 个刷新最好成绩：WikiText-2 困惑度 18.34（旧最好 39.14）、PTB 35.76（46.54）、LAMBADA 8.63（99.8）、LAMBADA 准确率 63.24（59.23）、CBT 常用名词 93.30、CBT 命名实体 89.05、enwik8 0.93 BPB、text8 0.98 BPC、WikiText-103 17.48；One Billion Word Benchmark 42.16 仍差于 21.8（Abstract、§3.1–§3.3、Table 3）[^src-gpt2-2019-language-models]。CoQA 零样本达 55 F1，未用 12.7 万条训练样本即匹敌或超过 4 个基线中的 3 个（Abstract、§3.5）[^src-gpt2-2019-language-models]。性能随容量近似 log-linear 提升（Fig 1、Fig 4）[^src-gpt2-2019-language-models]。

## 边界

模型在摘要任务上的表现「按定量指标仍只是 rudimentary」，实际可用性 far from use-able，且不少实际任务不优于随机（§6）[^src-gpt2-2019-language-models]；四个尺寸均 underfit WebText，留出困惑度随训练持续下降（§3）[^src-gpt2-2019-language-models]。WebText 与部分测试集存在文本重叠，以 8-gram 重叠率量化（§4）[^src-gpt2-2019-language-models]。模型改动与零样本表现之间未做消融，pre-LN 的贡献未单独分离。

[^src-gpt2-2019-language-models]: [[source-gpt2-2019-language-models]] — Radford et al., Language Models are Unsupervised Multitask Learners, OpenAI 技术报告, 2019
