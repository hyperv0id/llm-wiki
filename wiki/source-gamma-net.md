---
title: "Source: GAMMA-Net — GAT + Multi-Axis Mamba Interleaved for Traffic Forecasting"
type: source-summary
tags:
  - traffic-forecasting
  - mamba
  - graph-attention
  - state-space-model
  - spatial-temporal
  - arxiv-2026
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Source: GAMMA-Net

First Author et al., "GAMMA-Net — GAT + Multi-Axis Mamba Interleaved for Traffic Forecasting: Adaptive Long-Horizon Traffic ST Forecasting with Interleaved GAT + Multi-Axis Mamba", arXiv 2604.16859 (2026). Code: https://github.com/hdy6438/GAMMA-Net (not yet public).

## 核心动机

现有时空交通预测方法存在一个持久的三难困境：无法同时提供 (i) 高效长时记忆，(ii) 完全自适应的图推理，(iii) 适合实时 ITS 部署的轻量级计算[^src-gamma]。RNN-GNN 混合模型受限于循环块的梯度消失和平行化问题；Transformer 变体虽然能捕捉全局上下文，但 O(L²) 的复杂度在大图场景下难以承受；纯 Mamba 预测器虽然剥离了二次注意力，但空间线索仍隐含或绑定在固定邻接结构上——当传感器故障、车道关闭或拥堵波传播超出训练范围时，预测误差会急剧上升[^src-gamma]。

## 核心设计

GAMMA-Net 提出**交错式 GAT + 多轴 Mamba**架构来突破这个三难困境[^src-gamma]：

1. **Embedding Layer**：结合特征嵌入、周期嵌入（天/星期）和时空自适应嵌入，生成隐藏表示[^src-gamma]。
2. **GAT-Mamba Pair**：核心处理单元，以交错顺序排列：
   ```
   (GAT → Mamba_Temporal) × L → (GAT → Mamba_Spatial) × L
   ```
   其中 L=3[^src-gamma]。第一阶段 GAT 在每时间步动态重加权图结构，随后时间轴 Mamba 以线性复杂度捕捉长程时间依赖；第二阶段 GAT 基于时间上下文重新校准图注意力权重，随后空间轴 Mamba 在更新后的图中高效传播空间信息[^src-gamma]。这种交错的本质是：长程序列信息在每次图拓扑更新**之前**被蒸馏，反之亦然，形成一个闭环信息流[^src-gamma]。
3. **图结构**：基于物理路网直接连接构建边，使用预定义但非可学习的图拓扑作为结构基础[^src-gamma]。

## 实验结果

GAMMA-Net 在 6 个基准数据集（METR-LA、PEMS-BAY、PEMS03/04/07/08）上全面 SOTA，相比基线最多降低 **16.25% 的 MAE**[^src-gamma]。在 METR-LA 12 步（60 分钟）预测上 MAE 为 2.87（次优 STGM 3.23），RMSE 为 5.99（次优 7.02），MAPE 为 8.16%（次优 9.39%）[^src-gamma]。

消融实验揭示关键发现[^src-gamma]：
- 移除 GAT 使误差逐步上升（METR-LA 15min MAE +2.4%、60min 差距进一步扩大）
- 同时移除时间和空间 Mamba 扫描导致 **MAE 飙升 44%（METR-LA）和 45%（PEMS-BAY）**，证明双轴不可互换
- GAT → Mamba_Temporal → GAT → Mamba_Spatial 的顺序最优——每个组件为后续提供更新的拓扑先验

可视化分析[^src-gamma]：通过 SVD 分析 Mamba 状态转移矩阵的奇异值分布，证实空间组件集中捕获局部依赖，时间组件捕获更广泛的长程模式；通过 Louvain 社区检测展示了空间注意力忠实反映地理邻近性，时间注意力揭示功能性连接（行为相似但地理位置遥远的传感器）。

## 意义与局限

**贡献**：首次以交错式 GAT-Mamba 架构突破时空预测三难困境，实现动态图推理+线性复杂度长时记忆+轻量级部署的统一[^src-gamma]。

**局限**：依赖预定义图结构限制了对快速变化拓扑的适应性；深度模型的整体可解释性有限[^src-gamma]。论文未来方向包括动态图构建技术和跨领域扩展（天气预测、动态网络分析）[^src-gamma]。

[^src-gamma]: [[source-gamma-net]]
