---
title: "Object Queries（DETR 的可学习查询嵌入）"
type: technique
tags:
  - transformer
  - learned-queries
  - object-detection
  - non-autoregressive-decoding
created: 2026-08-30
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# Object Queries（DETR 的可学习查询嵌入）

**Object queries** 是 [[detr|DETR]] 论文对输入 Transformer decoder 的 N 个可学习位置编码（learned positional embeddings）的称呼：decoder 以这 N 个嵌入为输入，经自注意力与 encoder-decoder attention 逐层变换后，每个输出嵌入独立解码为一个预测（类别 + 框）或 ∅（no object）[^src-detr]。

## 为什么输入嵌入必须互异

论文指出 decoder 与 encoder 一样是置换不变的（permutation-invariant）：N 个输入嵌入必须互不相同才能产生不同的输出；论文将这组输入实现为可学习的位置编码（learned positional encodings），称为 object queries，并在每层注意力输入中加入（Sec 3.2）[^src-detr]。注意：置换不变性论证的是「输入必须互异」，「实现为可学习」是论文的设计选择——论文未对输出位置编码做「可学习 vs 固定」的消融，Table 3 中所有模型均使用 learned output positional encodings（Sec 4.2）[^src-detr]。论文的自注意力实现中，位置编码可学习或固定、但对给定序列在所有注意力层间共享（附录 A.1）[^src-detr]。

## 消融证据

- **必需性**：输出位置编码（object queries）必需、不能移除；实验只对比「仅在 decoder 输入传入一次」与「加到每层 decoder 注意力输入」两种方式（Sec 4.2, Table 3）[^src-detr]。
- **slot 特化**：将 20 个（共 N=100）query slot 在 COCO val 全集上的预测框中心可视化（Fig 7），各 slot 学到对特定区域与框尺寸的特化，且有多个操作模式；几乎所有 slot 都有预测全图大框的模式，作者假设这与 COCO 目标分布相关（Sec 4.3）[^src-detr]。
- **无强类别特化**：训练集中没有超过 13 只长颈鹿的图像，DETR 仍能在合成图中检出 24 只，论文以此说明 object query 不存在强类别特化（Fig 5, Sec 4.3）[^src-detr]。

## 在 wiki 谱系中的位置

wiki 层面的谱系整理：DETR 的「N 个可学习查询嵌入 → decoder 逐 query 并行解码」是后续把可学习 query 引入其他领域解码器设计（包括时空预测）的对照起点。相关的 query 机制设计：

- [[tqn]] / [[temporal-query-technique]] — 时序预测中用周期性偏移的可学习向量作为注意力 Query，注入全局先验；
- [[query-aggregate-attention]] — 时间 token 作为对固定空间结构基的查询，两步查询-聚合；
- [[generative-style-decoder]] / [[ar-vs-nar-decoding]] — 非自回归并行解码在时序预测中的对应决策。

注意：以上谱系定位是 wiki 的组织口径，上述论文与 DETR 论文均未互相声明继承关系。

## 相关页面

[[detr]] · [[source-detr]] · [[tqn]] · [[temporal-query-technique]] · [[query-aggregate-attention]] · [[ar-vs-nar-decoding]] · [[generative-style-decoder]]

[^src-detr]: [[source-detr]]
