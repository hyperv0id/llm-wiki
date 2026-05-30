---
title: "DSW Embedding (Dimension-Segment-Wise)"
type: technique
tags:
  - time-series
  - embedding
  - cross-dimension
  - segmentation
  - transformer
created: 2026-05-30
last_updated: 2026-05-30
source_count: 1
confidence: medium
status: active
---

# DSW Embedding (Dimension-Segment-Wise)

DSW embedding 是 [[crossformer|Crossformer]] 的核心嵌入方法，将输入 MTS 嵌入为 2D 向量阵列（时间轴 × 维度轴），使跨维度依赖可被显式建模 [^src-crossformer-2023]。

## 动机

先前 Transformer 模型将同时间步所有变量点嵌入为单一向量 $\mathbf{x}_t \in \mathbb{R}^D \to \mathbf{h}_t \in \mathbb{R}^{d_\text{model}}$，输出 $T$ 个向量的 1D 序列 [^src-crossformer-2023]。这有两个问题：
1. 单个数据点信息量少，不如时间片段有信息
2. 维度信息被压缩进向量内部，无法被注意力机制显式捕获

注意力分数图（Fig. 1a）显示 MTS 数据具有分段倾向——相近数据点有相似注意力权重，暗示片段比单点更适合作为 token [^src-crossformer-2023]。

## 方法

对输入 $\mathbf{x}_{1:T} \in \mathbb{R}^{T \times D}$，DSW embedding 分两步 [^src-crossformer-2023]：

**1. 分段**：每个变量的时间序列独立划分为长度 $L_\text{seg}$ 的段
$$\mathbf{x}_{i,d}^{(s)} = \{x_{t,d} \mid (i-1) \times L_\text{seg} < t \leq i \times L_\text{seg}\}$$

**2. 嵌入**：每段经线性投影 + 位置嵌入
$$\mathbf{h}_{i,d} = E\mathbf{x}_{i,d}^{(s)} + \mathbf{E}_{i,d}^{(\text{pos})}$$

输出为 2D 向量阵列 $\mathbf{H} = \{\mathbf{h}_{i,d}\}$，其中 $1 \leq i \leq T/L_\text{seg}$, $1 \leq d \leq D$。每个 $\mathbf{h}_{i,d}$ 代表**单变量**的一个时间段。

## 与其他嵌入方法的对比

| 方法 | 嵌入单位 | 输出结构 | 跨维度可建模性 |
|------|---------|---------|--------------|
| 传统 (Informer等) | 同时间步所有变量 → 1 向量 | 1D: $T$ 向量 | 隐式（压缩在向量内） |
| DSW (Crossformer) | 单变量时间段 → 1 向量 | 2D: $T/L_\text{seg} \times D$ 向量 | 显式（独立维度轴） |
| [[patch-based-tokenization|Patch Tokenization]] | 单变量 patch → 1 token | 1D: $N_\text{patch}$ per channel | 隐式/CI |

DSW 与 [[patch-based-tokenization|patch-based tokenization]] 类似（都是单变量分段），但 DSW 保留 2D 结构以支持跨维度注意力，而 patch tokenization 通常展平为 1D [^src-crossformer-2023]。

## 超参数 $L_\text{seg}$

段长 $L_\text{seg}$ 控制粒度 [^src-crossformer-2023]：
- 短期预测：小 $L_\text{seg}$ 更优
- 长期预测：大 $L_\text{seg}$ 更优（粗粒度减少序列长度）
- ETTh1 数据集上 $L_\text{seg}=24$ 恰好匹配日周期，效果最佳；48 则过粗

## Connections

- 属于：[[crossformer]] — 核心嵌入组件
- 对比：[[patch-based-tokenization]] — 类似分段但保留 2D 结构
- 相关：[[cross-dimension-dependency]] — DSW 保留维度信息使跨维度依赖可建模
- 后续：[[learnable-patch-position-encoding]] — CVPE 的位置编码也编码时间和变量维度

[^src-crossformer-2023]: [[source-crossformer-2023]]
