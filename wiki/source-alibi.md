---
title: "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
type: source-summary
tags:
  - transformer
  - position-embedding
  - extrapolation
  - iclr-2022
created: 2026-04-28
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# ALiBi 论文摘要

官方论文见 https://arxiv.org/abs/2108.12409，版本为 arXiv:2108.12409v2（ICLR 2022 正式版，2022-04-22，25 页），开源代码见 https://github.com/ofirpress/attention_with_linear_biases，本地文档 `raw/alibi-2108-12409.pdf`（`*.pdf` 被 `.gitignore` 排除，只保留本地副本；`pdftotext -layout` 提取文本 1492 行）。

## 核心论点

Transformer 各层函数本身与输入长度无关，推理时处理比训练更长序列失败源于位置表示方法[^src-alibi]。论文以 Baevski & Auli 语言模型为骨架，仅替换位置方法，并固定随机种子与 205 个 epoch 的其余超参做对照[^src-alibi]。

## 方法

ALiBi 全程不加位置向量，只在每个头 query-key 打分上加静态偏置 $m_h\cdot[-(i-1),\dots,-1,0]$。$m_h$ 训练前固定、不参与学习、不乘 $1/\sqrt{d_k}$ 缩放；$n$ 头时 $m_h=2^{-8h/n}$，序列右端恒为 $2^{-8}$[^src-alibi]。位置信息每层进入 Q、K，不进 V；实现上把偏置并入掩码，掩码由 $L\times L$ 变为 $n\times L\times L$[^src-alibi]。

## 证据

WikiText-103 上，$L=512$ 训练的模型在 $L_{valid}=3072$ 上为 18.40，低于 sinusoidal 在 $L=3072$ 训练的 $18.67\pm 0.24$，训练快 1.84 倍；$L=3072$ 的 ALiBi 为 17.60（Table 5、§4.1）[^src-alibi]。同 $L=3072$ 对比基线，ALiBi 为 17.60，T5 bias 为 18.01，rotary 为 18.57，sinusoidal 为 18.67；拉到 16072 token 时分别为 17.22、OOM、34.51、231.29（Table 4）[^src-alibi]。开销上，同 $L=512$ 训练为 28.3k 对 28.5k WPS、显存同为 15.3 GB（Table 1）；同长度下速度无损失，代价是 $n\times L\times L$ 掩码（100 MB 以内）[^src-alibi]。CC100+RoBERTa 1.3B 上，$L=512$ 的 ALiBi 与 $L=1024$ 的 sinusoidal 训练时长相同，$L_{valid}=1024$ 时为 9.30 对 9.24、显存 24.6 对 26.2 GB；$L_{valid}=2048$ 时为 9.54 对 48.85（Table 12）[^src-alibi]。斜率集合在 WikiText-103 上选定后原样用于 Toronto Books 与 1.3B 模型，未再调参（§4.1、A.3、§4.2）[^src-alibi]。

## 局限

滑窗（$S=1$）评估下困惑度随 $L_{valid}$ 持平，$L_{valid}>L$ 的收益主要来自缓解 early token curse，并未利用超过 $L$ 的上下文（Appendix B、Table 15）[^src-alibi]。峰值位置依设置而变：CC100 上约 $2L$，WikiText-103 上至少到 $3L$ 仍改善（§4.1、§4.2）[^src-alibi]。实验只验证三个语言建模语料，未测其他任务，也未与上下文扩窗类微调方法组合[^src-alibi]。

## 引用

[^src-alibi]: [[source-alibi]]
