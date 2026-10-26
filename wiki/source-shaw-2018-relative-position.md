---
title: "Self-Attention with Relative Position Representations"
type: source-summary
tags:
  - transformer
  - attention
  - position-encoding
  - relative-position
  - machine-translation
created: 2026-10-26
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# Self-Attention with Relative Position Representations

Peter Shaw、Jakob Uszkoreit、Ashish Vaswani（Google / Google Brain），arXiv:1803.02155v1，2018-03-06，5 页。

## 问题

Transformer 的自注意力机制对元素顺序本身具有置换不变性，位置信息依赖叠加在输入上的绝对正弦编码提供；这种编码将注意力打分固定在 $i$ 与 $j$ 各自的绝对位置上，所得表示无法随序列总长变化而保持不变[^src-shaw-2018-relative-position]。

## 方法

将输入序列视为标号有向全连接图，任意两元素之间存在边 $a_{ij}\in\mathbb{R}^{d_a}$。式 (3) 在 value 上加入 $a^V_{ij}$，式 (4) 在兼容性函数中加入 $a^K_{ij}$；在序列场景下边标签仅由截断相对偏移 $\operatorname{clip}(j-i,k)$ 决定，共 $2k+1$ 个，取 $d_a=d_z$。式 (5) 把打分拆解为内容项与位置项，位置项按位置 gather 边向量后通过 $n$ 组并行矩阵乘计算，步速约下降 7%[^src-shaw-2018-relative-position]。该工作形成两条可迁移结论：相对位置表示可以整体取代绝对位置编码；同一形式可作为将自注意力推广到标号有向图的接口[^src-shaw-2018-relative-position]。

## 证据

在 WMT 2014 翻译任务的测试集 newstest2014 上（Table 1），基线在同一代码库和相同配置下复现以隔离相对位置编码的贡献[^src-shaw-2018-relative-position]；base 模型中 EN-DE 的 BLEU 从 26.5 提升至 26.8、EN-FR 从 38.2 提升至 38.7，big 模型中 EN-DE 从 27.9 提升至 29.2、EN-FR 从 41.2 提升至 41.5。开发集消融实验（Table 3，base）显示：只保留 $a^K$ 与两者均保留的效果一致，BLEU 均为 25.8；去掉 $a^K$ 则降至 25.3，两者均不保留时仅为 12.5。Table 2 表明，$k\ge 2$ 后 BLEU 保持在 25.8–25.9，$k=0$ 为 12.5[^src-shaw-2018-relative-position]。

## 边界

论文仅在 EN-DE 和 EN-FR 两个翻译任务上进行验证，其中主表使用测试集，$k$ 敏感度分析与通路消融使用开发集 newstest2013。裁剪使「模型泛化到训练未见过的长度」属于动机陈述，论文未做外推实验；正文提及叠加绝对位置编码「未观察到进一步收益」，但未给出对应数据表格，Table 1 仅包含纯绝对与纯相对两种配置[^src-shaw-2018-relative-position]。

## 引用

[^src-shaw-2018-relative-position]: arXiv:1803.02155v1 全文，PDF 归档 `raw/shaw-2018-relative-position-1803-02155.pdf`。机制细节见 [[relative-position-representation]]，图推广见 [[relation-aware-self-attention]]。
