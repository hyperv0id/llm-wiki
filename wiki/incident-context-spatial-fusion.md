---
title: "Incident-Context Spatial Fusion (ICSF)"
type: technique
tags:
  - traffic-forecasting
  - attention-mechanism
  - spatial-fusion
  - incident-aware
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# Incident-Context Spatial Fusion (ICSF)

ICSF 是 [[igstgnn|IGSTGNN]] 的核心模块之一，负责捕获事件对交通网络的初始非均匀空间影响。核心思想：事件的影响取决于事件位置与传感器之间的距离、上下游关系和区域特征，因此不能均匀施加[^src-incident-guided-st-forecasting]。

## 问题动机

事件影响在空间上不均匀：
- 同一事件对不同传感器的冲击差异大，取决于距离和方向
- 现有方法（如简单 MLP 拼接）无法捕获事件-传感器-交通状态之间的复杂非线性关系
- 过度复杂的迭代消息传递（IMP）难以优化[^src-incident-guided-st-forecasting]

## 机制

### 1. 注意力组件投影

将交通隐藏状态 Hᵢ 和事件表示 I 投影到 Q/K/V 子空间：

- Q = HᵢW_Q（传感器当前状态）
- K = IW_K（事件属性匹配）
- V = IW_V（待聚合的事件信息）

### 2. 语义相关性 + 空间掩码

计算初始语义相关性 A_sem = KQ^T / √d_k，然后通过空间关系张量 D 进行掩码：

- 若事件 e_k 与传感器 v_j 在 D 中有连接 → 保留
- 否则 → 设为 -∞（注意力权重归零）

这一步强制执行拓扑先验：事件不应直接影响拓扑上无连接的远距离节点[^src-incident-guided-st-forecasting]。

### 3. 上下文感知融合

将初步注意力权重 α̃、传感器特征 S、空间关系 D 拼接后通过 MLP g_α + softmax 生成最终注意力权重 α：

α = softmax(g_α(α̃ ∥ S ∥ D))

这确保每个传感器根据自身区域特征获得定制化的事件影响表示[^src-incident-guided-st-forecasting]。

### 4. 加权聚合 + 残差融合

C = α^T V → 事件上下文向量

H'ᵢ = LayerNorm(Hᵢ + C) → 事件感知的更新表示

## 空间关系张量 D

D ∈ ℝ^{M×N×3}，为每个事件-传感器对构建 3 维向量：
1. **欧几里得距离** → 高斯核变换（捕获地理溢出效应）
2. **路网距离** → 高斯核变换（捕获沿道路的主线性传播）
3. **上下游关系** → 二值指标（区分上游严重影响 vs 下游可忽略影响）

上游方向的事件影响远大于下游方向[^src-incident-guided-st-forecasting]。

## 优势验证

IGSTGNN 中用 MLP 或 IMP 替换 ICSF 时，Alameda 上 MAE 分别退化 10.3% 和 12.5%，验证注意力机制在"捕获复杂关系"与"可优化性"之间取得最佳平衡[^src-incident-guided-st-forecasting]。

## 即插即用性

ICSF 可独立集成到其他 STGNN 骨干网络。在 AGCRN、GWNET、STTN、D2STGNN 上单独集成 ICSF 均带来一致 MAE 提升（Alameda 上 D2STGNN 提升 8.85%）[^src-incident-guided-st-forecasting]。

## 相关技术

- [[temporal-incident-impact-decay|TIID]] — 配套的时间衰减模块
- [[guided-layer-normalization|GLN]] — ConFormer 的条件归一化方法（不同路线解决类似问题）
- [[adaptive-graph-agent-attention|AGA-Att]] — FaST 的代理注意力机制（降低复杂度而非融合事件信息）

[^src-incident-guided-st-forecasting]: [[source-incident-guided-st-forecasting]]
