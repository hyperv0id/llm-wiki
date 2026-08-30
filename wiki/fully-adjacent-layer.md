---
title: "Fully-Adjacent Layer：打破 over-squashing 瓶颈的全邻接层"
type: technique
tags:
  - graph-neural-network
  - message-passing
  - long-range
  - iclr-2021
created: 2026-08-30
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# Fully-Adjacent Layer（全邻接层）

Fully-adjacent layer（FA 层）是 Alon & Yahav（ICLR 2021）为验证 over-squashing 瓶颈而引入的干预手段：对一个 K 层 GNN，把第 K 层改为「每一对节点之间都有边」的层，即该层内对每个节点取 $N_v := V$；不改变层类型、不增加权重，只修改单个层内的邻接结构（Sec 4.2）[^src-over-squashing]。前 K−1 层仍用原始稀疏图拓扑，只有第 K 层让已具备拓扑感知的节点表示直接两两交互（Sec 4.2）[^src-over-squashing]。

## 论文对方案自身的定位

论文明确说 FA 层是简单方案，其目的只是证明 over-squashing 在一些基准上普遍且未被处理到「连最简单的方案都有帮助」的程度；自述主要贡献是提出并解释 over-squashing 现象，而非方案本身（Sec 4.2）[^src-over-squashing]。实验设计上，作者沿用先前工作的原始代码与最优超参重训，不做额外调参，以排除调参作为改进来源（Sec 4.2）[^src-over-squashing]。

## 实验结果（作者报告口径）

- **QM9**（约 13 万个约 18 节点的分子图，回归 13 个量子化学性质；模型取 Brockschmidt (2020) 搜索 500 组配置后的最优设置，多为 K=8 层）：把最后一层改为 FA 层后，作者报告六种 GNN（R-GIN、R-GAT、GGNN、MLP、R-GCN、GNN-FiLM）的误差率平均相对降低 42%（Sec 4.2, Table 1, Table 4）[^src-over-squashing]。
- **NCI1 / ENZYMES**（Errica et al. (2020) 的公平比较实现，30 次测试运行 = 10 折 × 3 种子）：FA 层使误差率平均相对降低 4.8%（NCI1）与 12%（ENZYMES）；作者报告 NCI1 上 GIN+FA 比此前最佳的 GIN-base 高 1.5%，ENZYMES 上 GIN+FA 比 GIN-base 高 8.1%、比不使用图拓扑的 No Struct 基线高 2.5%（Sec 4.3, Table 2）[^src-over-squashing]。
- **VarMisuse**（程序变量误用预测，节点级任务，最佳模型用 6-10 层）：作者报告所有 GNN 加 FA 层后 SeenProjTest 准确率提升，取得新 SOTA 88.4%；UnseenProjTest 取得新 SOTA 83.8%（Sec 4.4, Table 3）[^src-over-squashing]。

## 归因与消融（Appendix B, Sec 4.2）

论文用一组对照回答「改进来自什么」：

- **不是 under-reaching 的缓解**：QM9 平均图直径 6.35±0.91、最大 10、90 分位 8，K=8 层下至少 90% 样本完全可达；加层到 10 层无改进（Sec 4.2）[^src-over-squashing]。
- **加大隐藏维度不是等效替代**：隐藏维度翻倍（128→256）只带来 5.5% 的误差下降，而同维度的 FA 层带来 43.4%（GCN, Table 5）[^src-over-squashing]。
- **图拓扑仍然必要**：把所有 K 层都改为 FA 层（等于忽略原图拓扑）误差反而比 base 高 1500%（+1520%, Table 5）；论文结论是节点直接交互须与拓扑感知叠加（Appendix B.2）[^src-over-squashing]。
- **层的放置**：倒数第二层为 FA 层、最后一层为普通 GNN 层的结果略好于最后一层 FA（−45.2% vs −43.4%）；在 FA 层上再叠一层 FA（2×FA）与单层 FA 相近（−43.3%）（Table 5）[^src-over-squashing]。
- **部分 FA**：随机采样最后一层 FA 边的一部分仍有效，误差下降与边比例相关：0.25× −8.4%、0.5× −31.5%、0.75× −37.1%、全量 −43.4%（GCN, Appendix B.3, Table 6）[^src-over-squashing]。

## 与其他绕开挤压手段的关系

论文在相关工作中指出，更早的工作已在事实上绕开长距离传播但未以 over-squashing 命名瓶颈：Gilmer et al. (2017) 加 virtual edges 缩短长距离、Scarselli et al. (2008) 加 supersource 节点、Allamanis et al. (2018) 以程序分析提供 16 种 shortcut 边类型（Sec 6）[^src-over-squashing]。论文未对 FA 层与这些手段做实验对比；本页不做优劣判断。

## 相关页面

- [[over-squashing]] — 本方案所针对的瓶颈概念与理论分析
- [[graphgps]] — 每层并行全局注意力分支提供节点间直接交互的模块化路线（对照设计）
- [[topology-aware-graph-transformer]] — 注意力矩阵内混合邻接与全局注意力（对照设计）

[^src-over-squashing]: [[source-over-squashing]]
