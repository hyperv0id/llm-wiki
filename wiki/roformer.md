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
last_updated: 2026-06-22
source_count: 1
confidence: medium
status: active
---

# RoFormer

**RoFormer**（Rotary Transformer）是由追一科技（Zhuiyi Technology）苏剑林团队提出的增强型 Transformer，使用 Rotary Position Embedding ([[rope|RoPE]]) 替代传统的绝对位置嵌入[^src-roformer]。

## 架构

RoFormer 本质上是对标准 Transformer 的一处关键修改：**将 self-attention 中的加法型位置编码替换为乘法型旋转变换**。具体而言：

- 原始 Transformer：$\boldsymbol{q}_m = W_q(\boldsymbol{x}_m + \boldsymbol{p}_m)$
- RoFormer：$\boldsymbol{q}_m = \boldsymbol{R}^d_{\Theta,m} W_q \boldsymbol{x}_m$

其中 $\boldsymbol{R}^d_{\Theta,m}$ 为 $d \times d$ 分块对角旋转矩阵，按 $m\theta_i$ 角度旋转每对维度[^src-roformer]。

## 变体

- **RoFormer (EN)**：基于 BERT-base-uncased 架构，英文预训练
- **RoFormer (ZH)**：基于 WoBERT（词粒度中文 BERT），中文预训练，特别适用于长文本任务

## 关键性能

| 任务 | 表现 |
|------|------|
| WMT 2014 EN-DE 翻译 | BLEU 27.5（略高于 Transformer 基线的 27.3） |
| BERT 预训练 | 比 BERT 收敛更快 |
| GLUE (MRPC) | F1 89.5（BERT: 88.9） |
| GLUE (QQP) | F1 86.4（BERT: 71.2，显著提升） |
| GLUE (STS-B) | Spearman 87.0（BERT: 85.8） |
| 中文 CAIL2019-SCM (1024长度) | 准确率 69.79%（WoBERT-512: 68.10%） |

## 生态

- **HuggingFace 集成**：已纳入 `transformers` 库 (`model_doc/roformer`)
- **GitHub**：https://github.com/ZhuiyiTechnology/roformer
- **作者**：苏剑林（Jianlin Su），追一科技，同时也是科学空间博客 (kexue.fm) 的作者

## 历史意义

RoFormer 论文提出的 RoPE 已成为现代大语言模型的事实标准位置编码方案。几乎所有主流开源 LLM（LLaMA、Mistral、Qwen、DeepSeek 等）均采用 RoPE 或其变体。RoFormer 本身作为一个 BERT 时代的模型并不算最突出，但其引入的 RoPE 技术影响深远[^src-roformer]。

## 相关页面

- [[source-roformer]] — 原始论文摘要
- [[rope]] — RoPE 技术详解
- [[siren-rope]] — 时间条件化 RoPE 扩展（LinkedIn, 2026）
- [[yarn]] — RoPE 上下文窗口扩展
- [[alibi]] — 替代位置编码方案

[^src-roformer]: [[source-roformer]]
