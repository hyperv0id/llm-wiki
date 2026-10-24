---
title: "Source: MiTA — Mixture of Top-k Attention (arXiv:2602.01219)"
type: source-summary
tags:
  - source
  - attention
  - efficient-attention
  - sparse-attention
  - mixture-of-experts
  - fast-weights
created: 2026-10-24
last_updated: 2026-10-24
source_count: 1
confidence: medium
status: active
---

# Source: MiTA（arXiv:2602.01219）

- **作者**：Qishuai Wen, Zhiyuan Huang, Xianghan Meng, Wei He, Chun-Guang Li（北京邮电大学人工智能学院）
- **时间**：2026-02-01；cs.LG / cs.CV；预印本，无接收 venue 信息
- **链接**：https://arxiv.org/abs/2602.01219
- **本地**：`raw/2602.01219-mita.md`（LaTeXML 转换的 HTML 全文，1272 行）
- **代码**：https://github.com/QishuaiWen/MiTA
- **题名不一致**：arXiv 页面题为 “Mixture-of-Top-$k$ Attention: Efficient Attention via Scalable Fast Weights”，正文题为 “MiTA Attention: Efficient Fast-Weight Scaling via a Mixture of Top-$k$ Activations”[^src-2602-01219-mita]。

## 核心论点

论文把 full attention 表示为隐藏宽度等于序列长度 $N$ 的两层 fast-weight MLP：权重为缩放后的 keys 与 values，偏置为零，激活为 $\exp$（式 3–4）。随上下文增长，MLP 宽度线性增加。论文将二次复杂度重述为「unbounded fast-weight scaling」，把高效注意力归纳为「扩展 fast weights」：压缩为窄模块，或将 query 路由至部分权重（式 1–4，§3.1）[^src-2602-01219-mita]。

论文提出五维分类法（缩放策略／专家数量／专家类型／专家构造／路由拓扑），将 Linear attention、Linformer、Agent attention、TTT、Deformable DETR、DAT、MoBA、NSA、Spark attention、DSA 归入统一坐标（Table 1）[^src-2602-01219-mita]。

## 方法：MiTA

MiTA 兼顾压缩与路由：对 $Q$ 分段平均池化得到 $m$ 个 landmark query（$m=25$），通过 $S^{kv}=K^{\top}\tilde Q/\sqrt d$ 一次打分，为每个 landmark query 选取 top-$k$ 个 KV（$k=25$）构成「形变专家」；分数经归一化得到 landmark values，与 landmark queries 组成常开的共享专家。每个 query 依 $Q^{\top}\tilde Q$ 的 argmax 路由至单个专家（$s=1$），拼接共享专家与目标专家的 key/value 执行单次 attention，通过 online softmax 合并（式 5–12，Algorithm 1）[^src-2602-01219-mita]。

## 证据

- ImageNet-1K 从头训练（Table 2）：MiTA-ViT-T 72.9 vs ViT-T 73.1，MiTA-ViT-S 78.5 vs ViT-S 77.9；MACs 1.1G / 4.4G（ViT 1.2G / 4.6G），参数量相同。
- ImageNet-21K 预训练权重微调 50 epoch（Table 3）：四个尺寸在 ViT 之后排第二（75.6 / 80.9 / 82.8 / 85.5 vs ViT 76.9 / 81.2 / 84.4 / 85.9）。
- ADE20K 分割，仅替换 Segmenter 解码器（Table 4）：Tiny 上 38.8 / 39.6 超过 38.1 / 38.8，Small 与 Base 上 ss 持平、ms 低 0.1，参数量少 1M / 3M / 10M。
- LRA（Table 5）：平均 59.26 vs full attention 59.37，Text 与 Retrieval 落后。
- 定性与效率（Fig. 4–8）：深层多数 token 未被选为专家（论文称 token-pruning 效应）；所选 KV 与路由 query 位置重叠 mIoU 偏低，论文用以说明 $s=1$ 为路由而非聚簇；RTX 4090 上测试序列长度至 $2^{20}$ 时的吞吐；$(m,k)$ 外推优于内插；MiTA 训练的 checkpoint 迁回 full attention 掉点较小。上述四项只有图示，无表格数值[^src-2602-01219-mita]。

## 局限

摘要自述「preliminary experiments on vision tasks」，优化与复杂场景留作后续；分类法自述「by no means complete」[^src-2602-01219-mita]。对照实验未包含 MoE attention（MoBA、NSA、VMoBA、MoGA）与 top-$k$ attention（Spark、DSA）基准，两项核心机制优势（形变专家 vs 刚性块、可复用 top-$k$ 集合 vs 逐 query top-$k$）只有分析论证；无 $m$、$k$、$s$ 在 MiTA 上的消融数据。gather 的随机访存被作者列为潜在显存带宽瓶颈，缓解方案引自 DSA 但未实测[^src-2602-01219-mita]。

机制细节、逐表数值与转换缺失项见 [[mita-attention]]，框架见 [[fast-weight-mlp-view-of-attention]]，分类法见 [[fast-weight-scaling-taxonomy]]。

[^src-2602-01219-mita]: [[source-2602-01219-mita]]
