---
title: "RoFormer"
type: entity
tags:
  - model
  - transformer
  - position-encoding
  - rotary-position-embedding
  - plm
  - zhuyi
created: 2026-06-22
last_updated: 2026-10-26
source_count: 3
confidence: medium
status: active
---

# RoFormer

**RoFormer**（Rotary Transformer）由追一科技苏剑林团队提出，把 self-attention 里的加法位置编码换成乘法旋转变换，即 Rotary Position Embedding（[[rope|RoPE]]）[^src-roformer]。

## 架构

对标准 Transformer 的改动只有一处：query 与 key 由 $W_q(x_m+p_m)$ 改为

$$q_m=R^d_{\Theta,m}W_qx_m,\qquad k_n=R^d_{\Theta,n}W_kx_n$$

$R^d_{\Theta,m}$ 是 $d\times d$ 分块对角正交矩阵，按 $m\theta_i$ 旋转第 $i$ 对相邻维度，$\theta_i=10000^{-2(i-1)/d}$[^src-roformer]。值向量不旋转，模型不新增位置参数，前向只多 $O(Nd)$ 的逐元素乘加。旋转保证内积只依赖相对位移 $n-m$，推导与数据流见 [[rope]]。

## 变体

RoFormer (EN) 以 `bert-base-uncased` 为基线，替换其正弦位置编码，预训练语料为 BookCorpus 与 Wikipedia（8:2 划分），batch 64、最大长度 512、100k 步，AdamW 学习率 $10^{-5}$[^src-roformer]；RoFormer (ZH) 以 WoBERT（词粒度中文 BERT）为基线替换绝对位置嵌入，预训练数据约 34GB（中文维基、新闻、论坛），分六阶段改变最大序列长度与 batch size，使模型适配不同长度[^src-roformer]。

## 实验

| 任务 | 设置 | RoFormer | 对照 | 出处 |
|------|------|----------|------|------|
| WMT 2014 EN-DE | fairseq，37k BPE，beam 4，5 checkpoint 平均 | BLEU 27.5 | Transformer-base 27.3 | 表 1 |
| GLUE MRPC (F1) | 3 epoch，max len 512，batch 32，lr $2,3,4,5\times10^{-5}$ | 89.5 | BERT 88.9 | 表 2 |
| GLUE STS-B (Spearman) | 同上 | 87.0 | BERT 85.8 | 表 2 |
| GLUE QQP (F1) | 同上 | 86.4 | BERT 71.2 | 表 2 |
| GLUE SST-2 (Acc) | 同上 | 90.7 | BERT 93.5 | 表 2 |
| GLUE QNLI (Acc) | 同上 | 88.0 | BERT 90.5 | 表 2 |
| GLUE MNLI (m/mm) | 同上 | 80.2 / 79.8 | BERT 84.6 / 83.4 | 表 2 |
| MLM 预训练损失 | BERT-base-uncased 对照 | 收敛更快 | — | 图 3 左 |
| Enwik8 语言建模 | Performer，12 层，768 维，12 头，字符级 | 同步数损失更低 | Performer 无 RoPE | 图 3 右 |
| CAIL2019-SCM，512 截断 | 中文预训练 + 相似案例匹配，验证/测试准确率 | 64.13 / 68.29 | WoBERT-512 64.07 / 68.10；BERT-512 64.13 / 67.77 | 表 5 |
| CAIL2019-SCM，1024 截断 | 同上 | 66.07 / 69.79 | 无同长度对照 | 表 5 |

表 2 的六项里 RoFormer 三升三降，论文表述为「在六项中的三项上明显优于 BERT」[^src-roformer]。CAIL2019-SCM 的 1024 长度结果中，正文称比 WoBERT 绝对提升 1.5%，表 5 测试列差值为 1.69、验证列为 2.00（66.07 对 64.07），两处口径并列[^src-roformer]。

中文多阶段预训练中，最大序列长度 1536 的第 5 阶段准确率最高（67.4%），128 的第 4 阶段最低（63.4%），论文以此说明序列长度上限与长文本能力的关系[^src-roformer]。

## 生态

作者为追一科技苏剑林（Jianlin Su），GitHub 地址为 https://github.com/ZhuiyiTechnology/roformer，HuggingFace 已并入 `transformers` 的 `model_doc/roformer`[^src-roformer]。

## 后续

RoPE 随后被广泛沿用与改造：上下文扩展将 Llama 2 的 4k 窗口扩至 128k（[[yarn|YaRN]]）[^src-yarn]，时间序列方向将固定序数索引换为时间条件化旋转（[[siren-rope]]）[^src-siren-rope]。论文对照均在 BERT-base 与 Transformer-base 量级，没有更大规模模型的横向比较[^src-roformer]。

## 相关页面

内容涵盖 [[source-roformer]] 原始论文摘要、[[rope]] RoPE 技术详解（推导、数据流、衰减界）、[[siren-rope]] 时间条件化 RoPE 扩展、[[yarn]] RoPE 上下文窗口扩展，以及 [[alibi]] 加性线性偏置的替代方案。

[^src-roformer]: [[source-roformer]]
[^src-yarn]: [[source-yarn]]
[^src-siren-rope]: [[source-siren-rope]]
