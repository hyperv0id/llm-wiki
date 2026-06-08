---
title: "Interleaved GAT-Mamba Architecture"
type: technique
tags:
  - traffic-forecasting
  - mamba
  - graph-attention
  - state-space-model
  - spatial-temporal
  - architecture-design
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Interleaved GAT-Mamba Architecture

**交错式 GAT-Mamba 架构**是 [[gamma-net|GAMMA-Net]] 的核心时空处理设计，将 Graph Attention Networks (GAT) 和 Mamba 选择性状态空间扫描以穿插交替的方式沿时空双轴堆叠[^src-gamma]：

```
(GAT → Mamba_Temporal) × L → (GAT → Mamba_Spatial) × L
```

其中 L 为堆叠层数（GAMMA-Net 中 L=3）[^src-gamma]。

## 动机

现有 GNN-Mamba 混合模型（如 STG-Mamba、SpoT-Mamba）存在一个共同缺陷：**时空推理一次性完成，时间理解与空间理解之间没有反馈**[^src-gamma]。这意味着新发现的时间模式不能重新塑造图结构，图结构的变化也不能指导下一轮时间推理。其结果是，当传感器故障、车道关闭或拥堵波模式变化时，预测误差剧烈上升。

交错架构通过**闭环信息流**解决这个缺陷：长程序列信息在每次图拓扑更新**之前**被蒸馏，而更新后的拓扑**立即**为下一轮时间推理提供条件[^src-gamma]。

## 设计原理

### 第一阶段：空间→时间（GAT → Mamba_Temporal）

1. **GAT** 在每个时间步 t 对输入 Z^(t) 进行多注意力头处理，动态重加权图边权重：
   ```
   H^(t) = GAT(G, Z^(t)),  t = 1, ..., T
   ```
   这使模型能根据当前交通状况过滤陈旧信号（如封闭匝道的影响）并放大的新兴模式（如溢出链路）[^src-gamma]。

2. **残差 + LayerNorm**：
   ```
   Ĥ = LayerNorm(H + Z)
   ```

3. **Mamba_Temporal** 沿时间轴扫描空间富集后的特征 Ĥ，以线性复杂度捕捉长程时间依赖：
   ```
   M = Mamba_Temporal(Ĥ)
   ```
   避免了 Transformer 的 O(L²) 代价和 RNN 的梯度消失瓶颈[^src-gamma]。

输出 Z_time 同时携带了**动态加权的空间信息**和**长程时间上下文**。

### 第二阶段：时间→空间（GAT → Mamba_Spatial）

4. **GAT** 基于 Z_time 重新校准图注意力权重：
   ```
   S = GAT(G, Z_time)
   ```
   这一轮 GAT 的输入已经被时间理解所丰富——新出现的时间模式（如中午突发拥堵）可以直接塑造空间图的连接强度[^src-gamma]。

5. **Mamba_Spatial** 沿空间轴扫描，在更新后的注意力调制图上传播信号：
   ```
   Z' = Mamba_Spatial(S)
   ```
   以线性复杂度实现高效的空间混合，避免了参数昂贵的全图卷积[^src-gamma]。

## 为什么这个顺序有效

消融实验揭示了每一步的必要性[^src-gamma]：

| 消融变体 | METR-LA 60min MAE | 变化 |
|----------|-------------------|------|
| 完整 GAMMA-Net | 2.87 | — |
| w/o GAT（移除两轮 GAT）| 2.94 | +2.4% |
| w/o Temporal（仅保留时间轴 Mamba）| 2.90 | +1.0% |
| w/o Spatial（仅保留空间轴 Mamba）| 2.88 | +0.3% |
| w/o Both（同时移除双轴 Mamba）| **4.14** | **+44%** |

关键解读[^src-gamma]：
- **GAT 不可少**：虽然底层路网图是固定的，但 GAT 的边缘重加权对于抑制过时影响和放大新兴模式至关重要。移除后误差随预测时长累积增长。
- **双轴不可互换**：时间 Mamba 和空间 Mamba 各司其职——前者压缩长序列为 memory-efficient 隐藏状态，后者将这些上下文丰富的信号沿图散布。仅保留一轴会断掉半边闭环，同时移除双轴则退化为浅层 GAT，性能崩溃。
- **顺序不可调换**：GAT 必须在每个 Mamba 扫描**之前**提供更新的拓扑先验，Mamba 输出必须**刷新节点特征**供下一轮 GAT 使用。若 GAT 保留但 Mamba 缺失，节点嵌入停止沿该维度演化，导致下一轮 GAT 面对陈旧特征。

## 与纯 Mamba 或纯 GAT 的区别

- **纯 Mamba 预测器**（如 S-Mamba 时空变体）：空间线索隐式，绑定在固定或随机游走图上 → 没有动态图重加权 → 拓扑变化时精度跳水[^src-gamma]
- **纯 GAT + RNN/CNN**（如 DCRNN, GWNet）：GAT 提供动态图推理，但时间建模受 RNN 串行瓶颈或 CNN 固定感受野限制 → 长程记忆能力不足[^src-gamma]
- **交错式 GAT-Mamba**：GAT 提供动态自适应图推理，Mamba 以线性复杂度提供长程记忆，闭环信息流确保时空理解相互增强[^src-gamma]

## 相关页面

- [[gamma-net|GAMMA-Net]] — 使用此架构的完整模型
- [[mamba|Mamba]] — 选择性状态空间模型
- [[s-mamba|S-Mamba]] — 首个 Mamba MTSF baseline
- [[dst-mamba|DST-Mamba]] — 时空分解 Mamba（AAAI 2025）
- [[stgcn|STGCN]] — 纯卷积图网络，图结构不可学习
- [[gwnet|GWNet]] — 自适应图学习，解决 STGCN 图不可学的局限

[^src-gamma]: [[source-gamma-net]]
