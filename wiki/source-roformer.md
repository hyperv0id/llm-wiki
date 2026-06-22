---
title: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
type: source-summary
tags:
  - position-encoding
  - transformer
  - nlp
  - attention
  - rotary-position-embedding
created: 2026-06-22
last_updated: 2026-06-22
source_count: 0
confidence: high
status: active
---

# RoFormer: Enhanced Transformer with Rotary Position Embedding

**Authors**: Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu (Zhuiyi Technology, 追一科技)
**arXiv**: 2104.09864v5, November 2023 (final version)
**原始标题**: RoFormer: Enhanced Transformer with Rotary Position Embedding

## 核心贡献

提出 **Rotary Position Embedding (RoPE)**，一种新的位置编码方法，通过旋转矩阵编码绝对位置，同时自然地在自注意力计算中纳入相对位置依赖。

## 关键思路

已有的位置编码方法（绝对位置嵌入、相对位置偏置）大多基于 **加法** 形式——将位置向量添加到上下文表示中。RoPE 的核心创新是采用 **乘法** 形式：

$$\langle f_q(\boldsymbol{x}_m, m), f_k(\boldsymbol{x}_n, n) \rangle = g(\boldsymbol{x}_m, \boldsymbol{x}_n, m-n)$$

即要求 query-key 内积仅通过相对位置 $(m-n)$ 编码位置信息。其解为复数旋转：

$$f_q(\boldsymbol{x}_m, m) = (W_q \boldsymbol{x}_m) e^{im\theta}$$
$$f_k(\boldsymbol{x}_n, n) = (W_k \boldsymbol{x}_n) e^{in\theta}$$

推广到 $d$ 维：将空间划分为 $d/2$ 个二维子空间，对每对维度施加角度为 $m\theta_i$ 的旋转，其中 $\theta_i = 10000^{-2(i-1)/d}$。

## RoPE 的性质

1. **长期衰减**：token 间内积随相对距离增大而衰减，符合语言直觉
2. **序列长度灵活**：可外推到训练时未见过的序列长度
3. **线性注意力兼容**：旋转保持向量范数不变，可直接与 Performer 等线性注意力结合
4. **与正弦编码的关系**：频率公式 $\theta_i$ 直接取自原始 Transformer 的正弦位置编码

## 实验验证

| 任务 | 基线 | RoFormer 表现 |
|------|------|---------------|
| WMT 2014 EN-DE 翻译 | Transformer (BLEU 27.3) | BLEU 27.5 |
| BERT 预训练 (MLM) | BERT | 更快收敛 |
| GLUE 微调 | BERT | MRPC 89.5 (+0.6)、QQP 86.4 (+15.2)、STS-B 87.0 (+1.2) |
| Performer + RoPE | Performer | 更快收敛 + 线性复杂度 |
| 中文长文本 (CAIL2019-SCM) | BERT/WoBERT | 1024 长度下 +1.5% 准确率 |

## 影响

RoPE 已成为大多数现代 LLM 的标准位置编码方案——LLaMA、Mistral、Qwen、DeepSeek 等均采用 RoPE。目前已有 YaRN（上下文扩展）、SIREN-RoPE（时间条件化）等众多衍生产作。

## 局限性（作者自述）

- 缺乏对"为什么旋转形式收敛更快"的彻底解释
- 虽证明了长期衰减性质，但缺乏对长文本表现优异的忠实解释

## 相关页面

- [[rope]] — RoPE 技术详解
- [[roformer]] — RoFormer 模型实体
- [[siren-rope]] — 时间条件化的 RoPE 扩展
- [[yarn]] — RoPE 上下文窗口扩展
- [[alibi]] — 线性偏置注意力（替代方案）
- [[generalized-positional-encoding-framework]] — 统一位置编码理论
