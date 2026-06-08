---
title: "GAMMA-Net"
type: entity
tags:
  - traffic-forecasting
  - mamba
  - graph-attention
  - state-space-model
  - spatial-temporal
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# GAMMA-Net

**GAMMA-Net** 是一个**交错式 GAT + 多轴 Mamba**时空交通预测模型，由 First Author 等于 arXiv 2026 提出。它通过将 Graph Attention Networks (GAT) 与选择性状态空间模型 (Mamba) 以交替方式沿时空双轴堆叠，首次在单一框架内同时实现动态图推理、线性复杂度长时记忆和轻量级部署[^src-gamma]。

## 核心架构

GAMMA-Net 由以下组件构成[^src-gamma]：

### 嵌入层

- **节点嵌入**：全连接层将原始特征（流量、时间戳、星期几）映射到高维空间
- **周期嵌入**：可学习的星期几和时间戳嵌入矩阵，编码周期模式
- **时空自适应嵌入**：借鉴 [[source-staeformer|STAEFormer]] 的设计，捕获时间邻近性关系

### GAT-Mamba Block

核心的交错处理序列[^src-gamma]：

```
(GAT → Mamba_Temporal) × L → (GAT → Mamba_Spatial) × L
```

- **第一阶段**（空间→时间）：GAT 在每时间步动态重加权图结构，随后时间轴 Mamba 以线性复杂度捕捉长程时间依赖
- **第二阶段**（时间→空间）：GAT 基于已更新的时间上下文重新校准图注意力权重，随后空间轴 Mamba 沿图拓扑高效传播空间信息
- L=3，每个 block 内都包含残差连接和 Layer Normalization

这构成了一个**闭环信息流**：时间理解在每次图重校准之前发生，反之图的重校准又为下一轮时间推理提供更新的拓扑先验[^src-gamma]。

### 回归层

将时空 GAMMA-Net 层的输出展平后通过全连接层生成最终预测[^src-gamma]。

## 性能与效率

### 基准实验

GAMMA-Net 在 6 个基准数据集上全面 SOTA[^src-gamma]：
- **METR-LA**（60min）：MAE 2.87（次优 STGM 3.23）、RMSE 5.99（次优 7.02）、MAPE 8.16%（次优 9.39%）
- **PEMS-BAY**（60min）：MAE 1.59（次优 STGM 1.86）、RMSE 3.67（次优 4.30）、MAPE 3.61%
- **PEMS03/04/07/08**：在 12 预测步长平均上全面领先，取得 **16.25% 的 MAE 最大降幅**

### 消融分析

- 移除两个 GAT 阶段 → METR-LA 15min MAE +2.4%、差距随预测时长进一步扩大[^src-gamma]
- 同时移除时间和空间 Mamba → MAE 飙升 **+44%（METR-LA）/+45%（PEMS-BAY）**，验证双轴不可互换[^src-gamma]
- 移除单轴（仅保留时间或仅保留空间）→ 误差适度上升，但远不如双轴移除严重[^src-gamma]

### 计算效率

Mamba 的线性复杂度使 GAMMA-Net 避免了 Transformer 的 O(L²) 代价，GAT 仅关注最相关邻居（而非全稠密图），两者结合使模型在 NVIDIA RTX 4090 上保持可部署性[^src-gamma]。

## 关键洞察

### 为什么是"GAT → Mamba_Temporal → GAT → Mamba_Spatial"？

这个顺序的精妙之处在于每步都在为下一步提供更新的先验[^src-gamma]：
1. GAT 根据当前交通状况动态重加权图 → 过滤已失效的影响（如封闭匝道）并放大新兴模式（如溢出链路）
2. Mamba_Temporal 以线性复杂度压缩长序列 → 将长时依赖蒸馏为 memory-efficient 的隐藏状态
3. GAT 利用时间上下文重新校准注意力 → 新出现的时间模式直接塑造空间图拓扑
4. Mamba_Spatial 沿更新后的图拓扑传播信号 → 避免全图卷积的参数膨胀

如果移除任何一环，闭环断裂，误差剧烈上升[^src-gamma]。

### 与现有 Mamba 交通模型的关系

现有工作如 [[dst-mamba|DST-Mamba]]、ST-MambaSync、STG-Mamba、SpoT-Mamba 等虽在部分方向上推进，但存在共同局限[^src-gamma]：
- 时空推理一次性完成 → 时间理解与空间理解之间无反馈
- 空间模块仍锚定在固定或随机游走图上 → 无法适应拓扑变化
- GAMMA-Net 通过交错闭环设计同时解决了这些问题

## 局限

- 依赖预定义图结构 → 当路网拓扑快速变化时适应性不足[^src-gamma]
- 黑箱可解释性 → 深度模型的通病，论文尝试通过 SVD/社区检测/t-SNE 等可视化提供部分洞察[^src-gamma]
- 代码尚未公开 → 论文声称将释放于 GitHub，目前不可用[^src-gamma]

## 相关页面

- [[source-gamma-net]] — 源文件摘要
- [[interleaved-gat-mamba]] — 交错式 GAT-Mamba 技术
- [[traffic-forecasting]] — 交通预测总览
- [[mamba]] — 选择性状态空间模型
- [[s-mamba]] — 首个 Mamba MTSF baseline（Neurocomputing 2024）
- [[dst-mamba]] — DST-Mamba，时空分解 Mamba（AAAI 2025）
- [[stgcn]] — STGCN，纯卷积时空图网络（IJCAI 2018）
- [[gwnet]] — GWNet，自适应图学习范式（IJCAI 2019）
- [[dcrnn]] — DCRNN，扩散卷积 RNN（ICLR 2018）

[^src-gamma]: [[source-gamma-net]]
