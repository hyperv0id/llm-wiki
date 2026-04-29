---
title: "FaST: Efficient and Effective Long-Horizon Forecasting for Large-Scale Spatial-Temporal Graph via Mixture-of-Expert"
type: source-summary
tags:
  - spatial-temporal-gnn
  - mixture-of-experts
  - long-horizon-forecasting
  - kdd-2026
created: 2026-04-29
last_updated: 2026-04-29
source_count: 0
references:
  - [[source-hyperd-hybrid-periodicity-decoupling]]
confidence: high
status: active
---

# FaST: Efficient and Effective Long-Horizon Forecasting for Large-Scale Spatial-Temporal Graph via Mixture-of-Expert

## 核心贡献

FaST 是一个针对大规模时空图（STG）长视野预测的高效且有效的框架，发表在 KDD 2026。论文提出两个关键创新：

1. **自适应图代理注意力（AGA-Att）**：使用少量可学习代理 token（a ≪ N）作为中介，在节点之间传递信息，将空间交互复杂度从 O(N²) 降低到 O(Na)。

2. **异质性感知 MoE（HA-MoE）**：用 Gated Linear Units（GLU）作为 experts 替换传统 FFN，实现高效可扩展的并行结构，同时避免 expert 极化问题。

## 关键发现

- 现有 STGNN 方法在处理大规模图和长视野预测时存在严重计算瓶颈：GNN 和注意力机制导致 O(N²) 和 O(T²) 复杂度
- 传统压缩方法（如 STID、CycleNet）采用"一刀切"方案忽视节点和时间段的异质性
- FaST 通过异质性感知路由器根据输入特征的空间-时间异质性动态选择专家避免 expert 极化
- GLU expert 相比 FFN expert 提升 1.4x 推理速度且保持精度

## 实验结果

在 LargeST 数据集（SD/GBA/GLA/CA，716-8600 节点）上：
- 672 步预测（1 周）MAE 提升 4.4%-18.4%
- 推理速度比 SOTA 快 1.3x-2.2x
- 内存复杂度与节点数线性相关可扩展到 8600 节点

## 与现有页面的关系

- 继承自 [[source-hyperd-hybrid-periodicity-decoupling|HyperD]] 的周期性建模思路但扩展到大规划图
- 与 [[traffic-forecasting]] 领域的 STGCN、DCRNN 等传统方法对比
- 作为 [[mixture-of-experts]] 在时空图预测中的具体应用

## 局限性

- 依赖预设图结构（可用预定义拓扑或数据自学习）
- 代理 token 数量 a 需要调优（a 太小可能丢失空间冗余信息）
- 在极小规模图（<100 节点）上优势不明显