---
title: "Longformer: The Long-Document Transformer"
type: source-summary
tags:
  - attention
  - sparse-attention
  - long-context
  - transformer
  - pretraining
created: 2026-10-26
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# Longformer: The Long-Document Transformer

Beltagy、Peters、Cohan（Allen Institute for Artificial Intelligence），arXiv:2004.05150v2（2020-12-02）。

问题：自注意力按 $O(n^2)$ 增长的时间与显存把预训练模型锁在 512 token 内；既有长序列 Transformer 集中在自回归语言建模，pretrain–finetune 的文档级迁移学习场景空缺（§1，Table 1）。

机制：注意力由滑动窗口、空洞滑动窗口与任务驱动 global attention 混合而成，整体 $O(n)$；global 位置与全序列对称计算注意力并使用独立投影 $Q_g,K_g,V_g$；打分提供 loop、chunk 与 CUDA kernel 三种实现，chunk 将打分显存从 $O(n^2)$ 降至 $O(nw)$（§3）。预训练从 RoBERTa checkpoint 继续 MLM：复制旧嵌入初始化新位置嵌入，在 Books、English Wikipedia、Realnews 与 Stories 子集合成的长文档语料上训练 65K 步、序列长 4,096（§5，Table 13）。另有 encoder–decoder 变体 LED，从 BART 初始化并将位置嵌入扩至 16K（§7）。

证据：字符级 LM 采用 5 个 phase 的分阶段训练（序列 2,048 翻倍到 23,040，窗口同倍增），小模型在 text8/enwik8 test 上 BPC 1.10/1.00，作者报告新 SOTA，大模型 enwik8 0.99 匹配 Sparse Transformer（Table 2、3）；窗口递增（32→512）优于固定与递减配置，2 头加 dilation 1.20 优于不加的 1.21（Table 4）；复制初始化使 MLM BPC 从 10.299 降到 1.957（Table 5）；六个下游任务 dev 一致超 RoBERTa-base（Table 7），WikiHop 81.9 与 TriviaQA 77.3 为提交时点 SOTA（Table 8）；WikiHop 消融显示 global attention 与独立投影缺一即大幅退化，无投影且无 global 时 65.5 对 73.8（Table 10）；LED 在 arXiv 摘要 R-1/R-2/R-L 46.63/19.62/41.83（Table 11）。

边界：global 位置需任务指定；dilation 与 RoBERTa 权重不兼容；chunk 实现不支持空洞、kernel 依赖 TVM；HotpotQA 落后当时最优 1 分（Table 9）。

## 相关页面

[[longformer-attention]] 覆盖三种模式的矩阵结构、依赖集合与维度推导。

[^src-beltagy-2020-longformer]: [[source-beltagy-2020-longformer]]
