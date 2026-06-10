---
title: "TEAM: Topological Evolution-aware Framework for Traffic Forecasting"
type: source-summary
tags:
  - spatiotemporal
  - traffic-forecasting
  - continual-learning
  - graph-evolution
  - dynamic-graph
created: 2026-06-10
last_updated: 2026-06-10
source_count: 0
confidence: high
status: active
---

# TEAM — Source Summary

**Authors**: Duc Kieu, Tung Kieu, Peng Han, Bin Yang, Christian S. Jensen, Bac Le
**Affiliations**: U. of Science HCM (Vietnam), Aalborg U (Denmark), UESTC (China), ECNU (China)
**Venue**: PVLDB 18(2): 265–278, 2024
**Code**: <https://github.com/kvmduc/TEAM-topo-evo-traffic-forecasting>

## 核心贡献

TEAM 是首个面向**演化道路网络（evolving RNs）**的交通预测框架。现有方法假设固定拓扑，需要每次拓扑变化时重新训练整个模型；TEAM 通过持续学习模块仅在新演化的部分上增量训练。

1. **问题形式化**：首次将演化 RN 上的交通预测定义为 graph snapshot 序列问题，每期图 ă_τ = ă_(τ−1) + Δă_τ。
2. **CAST 模型**：混合卷积+注意力的时空架构——ChebNetII 空间卷积 + GAT 空间注意力 + 扩张因果 TCN + 时间注意力。卷积捕获局部模式，注意力捕获全局模式，双层残差结构（forecast residual + backcast residual）。
3. **持续学习模块**：基于 Wasserstein 度量（EMD）的排练机制——计算每个节点在 RN 演化前后的数据直方图距离，选出最稳定节点（低 EMD → 巩固缓冲 Bc）和最不稳定节点（高 EMD → 更新缓冲 Bu），仅用缓冲+新节点数据训练。辅以 Elastic Weight Consolidation 正则化防止灾难性遗忘。
4. **复杂度**：O((ΔN + |B|)²) ≪ O(N²)，仅需训练演化部分。

## 实验结果

- PEMS03-Evolve（655→871 节点，7 个月）和 PEMS04-Evolve（180→248 节点）
- 对比 18 个基线（含 TrafficStream、EvolveGCN、DyRep 等动态图方法）
- 场景 1（全量重训）：CAST 精度最优
- 场景 2（持续学习）：TEAM 训练时间仅为 CAST 的 ~25%，精度竞争性
- 消融：移除持续模块导致 MAE 从 1.37 升至 4.71

## 局限

- 仅验证于 PEMS 数据集，其他城市/国家的泛化性待检验
- 持续学习模块的 EMD 计算需要额外预处理开销
- 对极快速演化（如每日变化）精度下降明显
