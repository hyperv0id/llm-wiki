---
title: "Query-Key Normalization for Transformers"
type: source-summary
tags:
  - transformer
  - attention
  - normalization
  - cosine-similarity
  - machine-translation
created: 2026-10-26
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# Query-Key Normalization for Transformers

Alex Henry、Prudhvi Raj Dachapally、Shubham Pawar、Yuxuan Chen（Cyndx Technologies），arXiv:2010.04245v1，2020-10-08，8 页，Findings of EMNLP 2020。

## 问题

缩放点积注意力用固定常数 $1/\sqrt{d}$ 把打分压入 softmax 可用区间，但点积本身无界：很小的相对差异就足以压制其余全部信号，限制注意力头能学到的模式复杂度[^src-henry-2020-qk-norm]。验证场景取低资源翻译。

## 方法

QKNORM 沿 head 维对每个 query 与 key 做 ℓ2 归一化，打分从内积转为余弦相似度，取值限制在 $[-1,1]$；因归一化后差值偏小，改乘可学习标量 $g$ 放大以代替除以 $\sqrt{d}$；$g$ 初始化为 $g_0=\log_2(L^2-L)$，$L$ 取训练集源/目标序列长度的 97.5 分位[^src-henry-2020-qk-norm]。它只作用于 Q 与 K，V 不归一，且与 LayerNorm 并用而非替代；相比 ScaleNorm（在切头前沿 embedding 维归一 Q、K、V，标量初值 $1/\sqrt{d}$），改动是将归一化移至切头之后、仅覆盖 Q 与 K，初值改为依赖序列长度[^src-henry-2020-qk-norm]。机制与 ℓ2 界的推导见 [[qk-norm]]。

## 证据

实验覆盖 5 个低资源方向：TED Talks 的 ar→en、sk→en、gl→en、en→he 与 IWSLT'15 的 en→vi，单卡训练 6–38 小时、参数 31M–48M（Table 1）[^src-henry-2020-qk-norm]。Table 2（Moses 脚本口径，bootstrap 重采样 $p<0.01$）相对 Nguyen & Salazar (2019) 平均增益 +0.928 BLEU；Table 3 用 SacreBLEU 复评，逐列差值平均 +0.99（本页按表计算，论文未报告）。增益集中在 gl→en 的 +2.20，其余四向为 +0.45 至 +0.68。另有一处独立演示：用 The Annotated Transformer 在 IWSLT 2016 de→en 训练 10 epoch，基线 19.4 BLEU、QKNORM 24.33 BLEU（SacreBLEU），热图显示注意力更分散，作者注明该对照「显然不系统」[^src-henry-2020-qk-norm]。

## 边界

对照模型与本方法有两处差异：基线用 ScaleNorm 与缩放点积注意力，本方法用 vanilla LayerNorm 与 QKNORM。平均增益因此同时包含两项改动，未做「基线保留 ScaleNorm、只换注意力打分」的消融（附录 Table 6 仅在 QK-Norm 内部逐项移除）[^src-henry-2020-qk-norm]。实验只覆盖低资源 NMT，未包含语言建模、句法任务或高资源设置；对饱和缓解与注意力分散的解释以 conjecture 给出，论文未测量 logit 幅度或注意力熵，并把「究竟在哪起作用、为什么起作用」列为待研究问题[^src-henry-2020-qk-norm]。

## 引用

[^src-henry-2020-qk-norm]: arXiv:2010.04245v1 全文（PDF 归档 `raw/query-key-normalization-for-transformers-2010-04245.pdf`）。机制、推导与边界展开见 [[qk-norm]]。
