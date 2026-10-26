---
title: "Attention Is All You Need"
type: source-summary
tags:
  - transformer
  - attention
  - machine-translation
  - encoder-decoder
created: 2026-10-26
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# Attention Is All You Need

该论文由 Ashish Vaswani、Noam Shazeer、Niki Parmar 等（Google Brain、Google Research、University of Toronto）发表于 NIPS 2017；对应 arXiv 版本为 arXiv:1706.03762v7（cs.CL，2023-08-02 版），官方地址为 https://arxiv.org/abs/1706.03762[^src-attention-1706-03762]。

## 问题

循环模型沿符号位置串行计算，同一样本内部无法并行，处理长序列时的批处理受显存限制；卷积方案虽支持并行，但任意两位置之间所需操作数随距离增长。论文的目标是仅依靠注意力机制建立输入与输出的全局依赖（§1、§2）[^src-attention-1706-03762]。

## 方法

编码器与解码器各设 6 层。编码器每层包含多头自注意力与逐位置前馈两个子层，各子层均先加入残差再进行层归一化；位置信息通过正弦函数叠加到嵌入向量上。编码器的自注意力为全连接结构，解码器的自注意力则通过上三角 $-\infty$ 掩码维持自回归特性，并通过第三个子层对编码器输出执行 cross-attention（§3.1–3.3）[^src-attention-1706-03762]。多头注意力将 512 维拆分至 8 个 64 维子空间并行计算后再进行拼接，使任意两位置间的路径长度降为常数，其代价是每层引入 $O(n^2 d)$ 复杂度以及注意力平均带来的分辨率损失（§2、§4）[^src-attention-1706-03762]。将点积除以 $\sqrt{d_k}$ 的原因在于，$d_k$ 增大会使点积方差按 $d_k$ 增长，从而将 softmax 推入梯度极小的区域（§3.2.1）[^src-attention-1706-03762]。

## 证据

在 WMT 2014 EN-DE 任务上，big 模型取得 28.4 BLEU，比当时最好的集成模型高 2.04；EN-FR 达到 41.8（Table 2），在 8 卡 P100 上训练 3.5 天；base 模型训练 12 小时、以 $3.3\times10^{18}$ FLOPs 达到 27.3/38.1（Table 2、§5.2）[^src-attention-1706-03762]。消融实验中，改用单头下降 0.9 BLEU，去掉 dropout 下降 1.2，去掉标签平滑下降 0.5，学习式位置嵌入为 25.7，正弦编码为 25.8（Table 3）[^src-attention-1706-03762]。

## 边界

实验验证限于 WMT 翻译与 WSJ 成分句法分析（F1 为 91.3 / 92.7，RNN Grammar 的 93.3 仍为最高）；受限邻域自注意力与向更长序列的外推均列为未来工作，未进行实验（§4、§6.3）[^src-attention-1706-03762]。EN-FR big 模型的 BLEU 在 §6.1 正文记为 41.0，Table 2 与摘要记为 41.8，两处数据并存。

---

[^src-attention-1706-03762]: [[source-attention-is-all-you-need-1706-03762]]
