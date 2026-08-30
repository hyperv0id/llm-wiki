---
title: "On the Bottleneck of Graph Neural Networks and its Practical Implications (Over-Squashing)"
type: source-summary
tags:
  - graph-neural-network
  - message-passing
  - long-range
  - bottleneck
  - iclr-2021
created: 2026-08-30
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# On the Bottleneck of Graph Neural Networks and its Practical Implications (Over-Squashing)

**Authors**: Uri Alon & Eran Yahav（Technion, Israel）
**Venue**: ICLR 2021（PDF 每页页眉「Published as a conference paper at ICLR 2021」，会议版式在 PDF 内核实）；对应 raw 文件 `raw/over-squashing-alon-2020.pdf`（水印 arXiv:2006.05205v4 [cs.LG], 2021-03-09）

## 核心论点

论文提出 over-squashing（信息挤压）概念，解释 GNN 难以传播远距离节点信息的现象：覆盖问题半径 r 需要 K ≥ r 层，而节点感受野随层数指数增长（论文引 Chen et al., 2018 为 O(exp(K))），式 (2)(3) 的求和把指数增长的信息压进固定长度向量，导致远端消息失效、模型只学到短程信号（Sec 1, Sec 3）[^src-over-squashing]。论文自述贡献是提出并论证该现象，不提出新 GNN 变体（Sec 1）[^src-over-squashing]。

## 证据（作者报告）

- Tree-Neighbors-Match 合成基准（Sec 4.1, Fig 3）：从 r=4 起 GCN 训练准确率仅 70%，r=5 起所有 GNN 无法完美拟合训练集；GCN/GIN 最多完美拟合 r=3，GAT/GGNN 达 r=4[^src-over-squashing]。
- 组合下界（Sec 5, Eq 4-5）：计数论证给出最小隐藏维度随 r 的下界，d=32 时最大 r=7；经验最小 d 更高（d=512 经验上最多拟合 r=7）。作者称 combinatorial lower bound，非定理-证明形式[^src-over-squashing]。
- 真实基准加 fully-adjacent layer（FA）打破瓶颈（Sec 4.2-4.4）：QM9 六种 GNN 误差平均相对降 42%（Table 1/4），NCI1 4.8%、ENZYMES 12%；VarMisuse SeenProj 88.4%、UnseenProj 83.8%（作者自述新 SOTA）。直径分析排除 under-reaching 解释（Sec 4.2）[^src-over-squashing]。
- 概念区分：与 under-reaching（信息不可达）和 over-smoothing（表示趋同）分立，两个构造性例子说明 over-smoothing 与 over-squashing 可独立发生（Sec 6, Appendix E）[^src-over-squashing]。

## 局限（论文自述）

长程解释为假设口径（Sec 1/4.1/6 的 we hypothesize）；短程问题明确不在范围内（Sec 3）；FA 层是演示性方案而非贡献主体（Sec 4.2）；组合下界分析仅对 GCN/GIN 成立，GAT 类上界至多加 1（Sec 5 脚注 1）[^src-over-squashing]。

## 相关页面

- [[over-squashing]] — 概念页
- [[fully-adjacent-layer]] — FA 层技术页

[^src-over-squashing]: [[source-over-squashing]]
