---
title: "Rethinking Attention with Performers (Choromanski et al., ICLR 2021)"
type: source-summary
tags:
  - transformer
  - linear-attention
  - kernel-methods
  - random-features
  - efficiency
created: 2026-08-30
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# Rethinking Attention with Performers

**作者 / 发表**：Krzysztof Choromanski、Valerii Likhosherstov、David Dohan、Xingyou Song、Andreea Gane 等 13 人（Google / University of Cambridge / DeepMind / Alan Turing Institute，前八人共同一作，作者行以 ∗ 标注 Equal contribution）。PDF 每页页眉为 "Published as a conference paper at ICLR 2021"，会议标识在 PDF 内核实；首页水印 arXiv:2009.14794v4 [cs.LG] 19 Nov 2022。raw 文件：raw/performer-choromanski-2020.pdf[^src-performer]。

## 核心论点

论文提出 Performer，用 FAVOR+（Fast Attention Via positive Orthogonal Random features）机制以线性空间与时间复杂度估计常规 softmax 全秩注意力，不依赖稀疏或低秩先验；论文自述这是首个对 softmax 全秩注意力具备可证明精度估计的 Transformer 架构（摘要、Sec 1）[^src-performer]。

## 方法

核化注意力 A(i,j)=K(qᵢ,kⱼ)、K(x,y)=E[φ(x)ᵀφ(y)]，经结合律按 Q′((K′)ᵀV) 计算，时间 O(Lrd)、空间 O(Lr+Ld+rd)（Sec 2.2）[^src-performer]。OR+ 部分解决核估计：sin/cos 三角特征虽无偏，但核值趋 0 时 MSE 发散、负特征值破坏归一化（Lemma 2）；论文提出正随机特征 PRF，无偏且 MSE 随核值趋 0 而趋 0（Lemma 1/2）；正交随机特征 ORF 在任意维度 d 降低 MSE 并给出更小的指数尾界（Theorems 2/3）；SMREG 正则化核是 softmax 核的通用下界（Theorem 1）；投影数 m 只依赖 d、ε、R，不依赖序列长度 L（Theorem 4）。因果注意力用 prefix-sum 实现（Appendix B.1）[^src-performer]。

## 实验

作者报告：V100 上反向传播近线性时间、次二次内存，接近注意力直接返回 V 的理论上限（Fig 3）；正交与正特征显著降低近似 MSE（Fig 4）；PG-19 上三角特征不稳定，正特征加重采样才匹配常规 Transformer（Fig 5）；36 层 TrEMBL 上 Reformer/Linformer 显著掉点，ReLU 核 Performer test 准确率 (U) 31.58/(B) 36.09 vs Transformer 30.80/33.32（Table 2）；ImageNet64（L=12288）上 Performer/6 层匹配 Reformer/12 层（Fig 7）；附录转引 LRA，作者报告在速度 >100 examples/sec 的可扩展方法中分数最高（Fig 19）[^src-performer]。

## 局限

固定 m 无法近似无限长序列的 hard attention，m 依赖 d、ε、R（Appendix F.6）；近似误差可被多层网络放大，预训练权重迁移需小量微调（Sec 4.3）；ORF 要求 m ≤ d（Sec 2.4）[^src-performer]。

[^src-performer]: [[source-performer]]
