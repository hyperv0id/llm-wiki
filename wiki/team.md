---
title: "TEAM"
type: entity
tags:
  - spatiotemporal
  - traffic-forecasting
  - continual-learning
  - graph-evolution
created: 2026-06-10
last_updated: 2026-06-10
source_count: 1
confidence: high
status: active
---

# TEAM — Topological Evolution-aware Framework

TEAM 是首个面向演化道路网络（evolving RNs）的交通预测框架（PVLDB 2024）[^src-team]。与现有方法假设固定拓扑不同，TEAM 通过持续学习机制增量适应节点的增删和拓扑变化，大幅降低再训练成本。

## 动机

城市化导致道路网络持续演化——新路修建、旧路废弃、传感器增减。传统交通预测模型依赖固定拓扑，每次 RN 变化都需要从零重新训练，计算代价高昂[^src-team]。简单的模型迁移面临两个挑战：(1) 历史数据与新数据的模式可能不一致（distribution shift）；(2) 有用知识在迁移后被遗忘（catastrophic forgetting）。

## 架构组成

### CAST（核心预测模型）

混合卷积+注意力架构：
- **空间组件**：ChebNetII 图卷积 → GAT 多头空间注意力
- **时间组件**：扩张因果 TCN → 时间自注意力
- **ST Block**：每 block 含空间→时间→forecast/backcast 卷积
- **ST Stack**：多个 block 通过双层残差连接（forecast residual + backcast residual）
- 设计理念：先卷积（局部特征）后注意力（全局记忆），训练更高效

### 持续学习模块

核心流程：
1. 取每个节点在 RN 演化前后各 τ 个时间步的数据，构建直方图
2. 计算 Earth Mover's Distance (EMD) 量化节点稳定性
3. 低 EMD 节点 → 巩固缓冲 Bc（排练用）
4. 高 EMD 节点 → 更新缓冲 Bu（重新学习用）
5. 仅用新节点 + 缓冲区数据训练模型
6. EWC 正则化保护重要参数不被覆盖

## 关键结果

| 指标 | 场景 1（全量重训 CAST） | 场景 2（持续 TEAM） |
|------|----------------------|-------------------|
| PEMS04 MAE | 1.28 | 1.37 |
| 训练时间 (PEMS03) | ~27,000s | ~6,700s (4× 加速) |
| 对比 TrafficStream | — | 精度更高 + 支持收缩 |

[^src-team]: [[source-team]]
