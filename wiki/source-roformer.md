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
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# RoFormer: Enhanced Transformer with Rotary Position Embedding

**Authors**: Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu（追一科技）。**arXiv**: 2104.09864。v1 2021-04-20，v2 2021-10-09，v3 2022-08-05，v4 2022-08-09，v5 2023-11-08；本页与 `raw/` 中的 PDF 均对应 v5。v1 仅给出中文数据初步结果并注明英文基准待补，v5 补入 WMT 翻译、掩码语言建模、GLUE 与 Performer 四组实验，作者名单增至六人。

## 核心论点

自注意力打分与 token 排列无关，位置信息须由外部注入。既有方案将位置项加在词嵌入或注意力分数上，位置与内容在加法中混合，无法适配线性注意力架构[^src-roformer]。论文要求内积只依赖相对位移 $m-n$，给出乘法形式的解。

## 机制

二维解为 $f_q(x_m,m)=(W_qx_m)e^{\mathrm im\theta}$：模长不变，辐角随位置等差增长。$d$ 维将相邻两维配为 $d/2$ 对子空间，第 $i$ 对旋转 $m\theta_i$，$\theta_i=10000^{-2(i-1)/d}$ 与正弦位置编码同源；旋转矩阵分块对角且正交，内积化为 $x_m^\top W_q^\top R^d_{\Theta,n-m}W_kx_n$，绝对位置在 $(R^d_{\Theta,m})^\top R^d_{\Theta,n}$ 中相消；实现以逐元素乘加代替矩阵乘，代价 $O(Nd)$，无新增参数。长距离衰减证明将内积按通道对写成复数求和，经 Abel 变换后模的上界分解为 $\max_i|h_{i+1}-h_i|\sum_i|S_{i+1}|$，相位部分随相对位移增大而下降（图 2）。旋转保范，位置乘在非线性特征映射之后，线性注意力仍保持相对位置形式；该写法保留未旋转的分母以防除零，分子允许出现负项。

## 证据

WMT 2014 EN-DE：BLEU 27.5 对 Transformer-base 27.3（表 1）。与 BERT-base 对照的掩码语言建模损失下降更快（图 3 左）。GLUE 六项三升三降：MRPC 89.5/88.9、STS-B 87.0/85.8、QQP 86.4/71.2 上升，SST-2 90.7/93.5、QNLI 88.0/90.5、MNLI 80.2/79.8 对 84.6/83.4 下降（表 2）。Performer 换用 RoPE 后 Enwik8 损失更低（图 3 右）。CAIL2019-SCM：RoFormer-1024 测试 69.79%，RoFormer-512 68.29%，WoBERT-512 68.10%（表 5）。

## 局限

§4.5.5 列出两条：未解释收敛更快的原因；未解释长文本优势成因，仅说明长期衰减与既有位置编码相似。

## 相关页面

包含 [[rope]]（推导、数据流与衰减界）、[[roformer]]（RoFormer 模型实体）、[[siren-rope]]（时间条件化的 RoPE 扩展）、[[yarn]]（RoPE 上下文窗口扩展）、[[alibi]]（线性偏置注意力替代方案）与 [[generalized-positional-encoding-framework]]（统一位置编码理论）。

[^src-roformer]: [[source-roformer]]
