---
title: "Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval"
type: source-summary
tags:
  - time-series-forecasting
  - periodicity
  - global-retrieval
  - plug-and-play
  - multivariate
created: 2026-05-31
last_updated: 2026-05-31
source_count: 0
confidence: medium
status: active
---

# Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval

**Authors:** Fanpu Cao, Lu Dai, Jindong Han, Hui Xiong (HKUST-GZ / HKUST / Shandong University)
**Venue:** ICLR 2026
**Code:** <https://github.com/macovaseas/GTR>

## 核心论点

现有 MTSF 模型受限于固定回溯窗口，无法捕获超过窗口长度的全局周期模式。直接扩展窗口导致过拟合、计算成本和冗余信息处理。GTR 通过维护整个周期长度的可学习全局时间嵌入，根据绝对位置检索对应段并与局部输入对齐，以 2D 卷积和残差连接融合全局与局部信息，实现即插即用的周期感知增强。

## 方法

GTR 包含三个关键步骤：(1) 根据输入序列的绝对时间位置（t₀ mod L），从可学习的全局参数矩阵 Q ∈ R^(L×N) 中检索对应的周期参考向量 qₙ；(2) 将输入序列与检索到的全局参考堆叠为 2×T 矩阵，通过 2D 卷积核（宽度 1+2⌊P/2⌋，P 为高频局周期长度）提取跨尺度时间模式；(3) 通过残差连接将增强特征与原始输入融合后送入主干模型。核心公式：zₙ = xₙ + Dropout(C([xₙ; qₙ]; κ=(2, 1+2⌊P/2⌋)))。

主干模型采用轻量 MLP（GeLU 激活，残差连接），并使用 RevIN 处理分布偏移。GTR 本身仅 40.1K 参数，完整系统 0.98M 参数，为 iTransformer 的 19%。

## 关键发现

1. **全局周期相关性更强**：Electricity 数据集上的 Pearson 相关性分析表明，跨全局周期的片段相关性（Corr(S₁₂, S₅)=0.96）高于邻近片段（Corr(S₁₂, S₁₃)=0.94）。
2. **短窗口优势显著**：GTR 在最短回溯窗口下表现最优，基线模型在窗口减少时 MSE 指数增长，GTR 保持稳定。
3. **跨模型泛化强**：GTR 可提升 iTransformer（PEMS03 上 MSE 降低 62.2%）、PatchTST（PEMS04 降低 56.2%）、DLinear（PEMS04 降低 91.9%）。
4. **理论支撑**：贝叶斯估计框架证明，在 σ²_ε<σ²_η 条件下，GTR 融合后的变量间相关性估计误差严格小于原始观测的估计误差。

## 局限

固定全局周期长度假设不适用于变化周期的数据；跨通道共享周期长度不适用于异质周期的多变量场景；长周期时 2D 卷积核宽度线性增长带来计算负担；输入序列长时线性投影层 O(NT²) 复杂度成为瓶颈。代码公开，未使用 LLM 辅助写作。
